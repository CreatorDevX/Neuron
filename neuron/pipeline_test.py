"""Run a short real-data training pipeline and summarize measured throughput.

Example:
  python -m neuron.pipeline_test --config configs/neuron-100m-t4.yaml --steps 5 \\
      --set train.micro_batch=4 train.grad_accum=1 model.loss_chunk=128

The run uses the normal trainer, loader, model, optimizer, precision, compile,
logging, and checkpoint paths. It requires prepared data by default; pass
--allow-auto-prepare to permit the regular Hugging Face bootstrap behavior.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="neuron.pipeline_test",
        description="Run a few real training steps and report cold/steady throughput.",
    )
    parser.add_argument("--config", default="configs/kaggle-2xt4.yaml")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--allow-auto-prepare", action="store_true",
                        help="allow downloading/preparing data if local shards are absent")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides",
                        metavar="section.key=value",
                        help="override any model or train config field; repeatable values")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be at least 1")

    # Share the default directory across torchrun ranks. An explicit --out-dir
    # is recommended when launching distributed without TORCHELASTIC_RUN_ID.
    run_id = os.environ.get("TORCHELASTIC_RUN_ID")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not args.out_dir and (
        not run_id or run_id.lower() == "none"
    ):
        parser.error("pass --out-dir for distributed runs when TORCHELASTIC_RUN_ID is unavailable")
    if not run_id or run_id.lower() == "none":
        run_id = str(time.time_ns())
    out_dir = args.out_dir or os.path.join("runs", f"pipeline-test-{run_id}")

    from .train import main as train_main

    train_args = [
        "--config", args.config,
        "--max-steps", str(args.steps),
        "--resume", "none",
        "--out-dir", out_dir,
        "--disable-wandb",
    ]
    if args.data_dir:
        train_args.extend(["--data-dir", args.data_dir])
    if not args.allow_auto_prepare:
        # A short throughput check should not unexpectedly start multi-hour prep.
        train_args.append("--no-auto-prepare")
    defaults = [
        "train.log_every=1",
        "train.eval_every=0",
        "train.ckpt_every_steps=0",
        "train.ckpt_every_min=0",
    ]
    train_args.extend(["--set", *defaults, *args.overrides])
    train_main(train_args)

    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return
    log_path = Path(out_dir) / "log.jsonl"
    records = []
    if log_path.exists():
        with log_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "tok_s" in row and "step_s" in row:
                    records.append(row)
    if not records:
        print(f"[pipeline-test] no throughput records found in {log_path}", flush=True)
        return

    # Exclude the first step from steady-state throughput because it commonly
    # includes torch.compile's initial specialization/compilation cost.
    steady = records[1:] if len(records) > 1 else records
    median_tok_s = statistics.median(float(row["tok_s"]) for row in steady)
    median_step_s = statistics.median(float(row["step_s"]) for row in steady)
    cold = records[0]
    print(
        f"[pipeline-test] completed through step {int(records[-1]['step'])}; output={out_dir}\n"
        f"[pipeline-test] first-step {float(cold['tok_s']):,.0f} tok/s, "
        f"{float(cold['step_s']):.3f} s/step\n"
        f"[pipeline-test] steady median {median_tok_s:,.0f} tok/s, "
        f"{median_step_s:.3f} s/step; peak VRAM {max(float(r.get('mem_gb', 0)) for r in records):.2f} GB",
        flush=True,
    )


if __name__ == "__main__":
    main()
