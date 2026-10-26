"""Prepare and train an HF stream in resumable, step-sized chunks.

Example: python -m neuron.stream_train --config configs/colab-1xt4.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from .checkpoint import read_latest
from .config import build_configs, world_tokens_per_step
from .data import load_manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="defaults to the Colab or Kaggle profile based on visible GPUs")
    ap.add_argument("--spec", default="configs/data_spec.yaml")
    ap.add_argument("--steps-per-chunk", type=int, default=2000,
                    help="number of next optimizer steps to provision at a time")
    ap.add_argument("--resume", default="auto")
    ap.add_argument("--disable-wandb", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = ap.parse_args()
    if args.steps_per_chunk < 1:
        ap.error("--steps-per-chunk must be positive")

    world = max(1, torch.cuda.device_count())
    config_path = args.config or ("configs/kaggle-2xt4.yaml" if world >= 2
                                  else "configs/colab-1xt4.yaml")
    mcfg, tcfg = build_configs(config_path, overrides=args.overrides)
    tps = world_tokens_per_step(tcfg, world, mcfg)
    schedule_steps = max(1, math.ceil(tcfg.total_tokens / tps))
    chunk_tokens = tps * args.steps_per_chunk
    data_dir = Path(tcfg.data_dir)
    progress_path = data_dir / "progress.json"
    with open(args.spec, encoding="utf-8") as spec_file:
        source_specs = (yaml.safe_load(spec_file) or {}).get("sources", [])
    source_ids = [source.get("id", f"src{i}") for i, source in enumerate(source_specs)]
    try:
        from .hub import sync_data
        sync_data(tcfg.data_dir, download=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[hub][info] data restore skipped: {exc}", flush=True)
    if args.resume == "auto" and read_latest(tcfg.out_dir) is None:
        try:
            from .hub import restore_latest
            restore_latest(tcfg.out_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[hub][info] checkpoint restore skipped: {exc}", flush=True)
    exhausted = False

    while True:
        pointer = read_latest(tcfg.out_dir)
        step = pointer[0] if pointer else 0
        tokens_seen = step * tps
        available = 0
        if (data_dir / "manifest.json").is_file():
            available = int(load_manifest(str(data_dir)).get("totals", {}).get("train", 0))
        pending = max(0, available - tokens_seen)

        progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
        exhausted = bool(source_ids) and all(
            progress.get(source_id, {}).get("exhausted", False) for source_id in source_ids
        )
        if pending < chunk_tokens and not exhausted:
            need = max(1, chunk_tokens - pending)
            print(f"[stream] step={step:,} tokens_seen={tokens_seen:,} available={available:,}; "
                  f"preparing about {need:,} more train tokens", flush=True)
            subprocess.run([
                sys.executable, "-m", "neuron.prepare_data", "--spec", args.spec,
                "--out", tcfg.data_dir, "--tokenizer", tcfg.tokenizer_path,
                "--max-new-tokens", str(need),
            ], check=True)
            available = int(load_manifest(str(data_dir)).get("totals", {}).get("train", 0))
            pending = max(0, available - tokens_seen)
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            exhausted = bool(source_ids) and all(
                progress.get(source_id, {}).get("exhausted", False) for source_id in source_ids
            )

        if pending <= 0:
            print("[stream] no untrained tokens remain", flush=True)
            break

        steps = min(args.steps_per_chunk, max(1, math.ceil(pending / tps)))
        target_step = step + steps
        docs_processed = sum(progress.get(source_id, {}).get("train", {}).get("docs", 0)
                             for source_id in source_ids)
        print(f"[stream] dataset docs={docs_processed:,} "
              f"tokens={available:,}; training steps {step:,}..{target_step:,}", flush=True)
        command = [
            sys.executable, "-m", "neuron.train", "--config", config_path,
            "--spec", args.spec, "--no-auto-prepare", "--stream-chunk",
            "--resume", args.resume, "--max-steps", str(target_step),
            "--schedule-steps", str(schedule_steps),
        ]
        if args.disable_wandb:
            command.append("--disable-wandb")
        if args.overrides:
            command.extend(["--set", *args.overrides])
        if world > 1:
            command = ["torchrun", "--standalone", f"--nproc_per_node={world}",
                       "--module", "neuron.train", *command[3:]]
        subprocess.run(command, check=True)

        if exhausted and pending <= chunk_tokens:
            print(f"[stream] source exhausted at {available:,} train tokens; training complete", flush=True)
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
