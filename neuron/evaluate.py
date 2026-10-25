"""Held-out loss / perplexity for a checkpoint.

  python -m neuron.evaluate --ckpt runs/neuron-100m/final.pt --tokens 4000000

Streaming over the `val` split with the checkpoint's own model config; batch
contiguous blocks; reports loss, ppl, tok/s. Val stream is read without
shuffle (stable slice) starting at `--start-token` (default 0) so different
checkpoint evals are directly comparable.
"""
from __future__ import annotations

import argparse
import contextlib
import math
import os
import time

import torch

from .config import ModelConfig
from .checkpoint import build_run_identity
from .data import TokenDataset
from .model import GPT
from .train import micro_batches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", default="val")
    ap.add_argument("--tokens", type=int, default=4_000_000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--start-token", type=int, default=0)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--allow-different-data", action="store_true",
                    help="evaluate on a dataset/tokenizer different from the checkpoint inputs")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    identity = state.get("identity")
    if identity and not args.allow_different_data:
        train_cfg = state.get("train_cfg") or {}
        current_identity = build_run_identity(args.data_dir, train_cfg.get("tokenizer_path", ""))
        if current_identity != identity:
            raise SystemExit("evaluation data/tokenizer identity differs from the checkpoint; pass --allow-different-data to evaluate it intentionally")
    mcfg = ModelConfig(**state["model_cfg"])
    model = GPT(mcfg)
    model.load_state_dict(state["model"], strict=True)
    model.set_gradient_checkpointing(False)
    model.to(device).eval()
    dtype = torch.float32 if (args.fp32 or device.type == "cpu") else (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

    ds = TokenDataset(args.data_dir, args.split, seq_len=mcfg.seq_len,
                      shuffle=False, seed=0)
    if ds.stream.total < args.tokens:
        raise SystemExit(f"val split has {ds.stream.total:,} tokens < requested {args.tokens:,}")
    blocks = (args.tokens + mcfg.seq_len - 1) // mcfg.seq_len

    # deterministic slice starting at block index from start-token
    from .data import make_loader

    start_i = args.start_token // mcfg.seq_len
    loader = make_loader(ds, start=start_i, count=blocks, num_workers=2, prefetch=4, pin=True)

    make_acx = (lambda: torch.autocast(device_type="cuda", dtype=dtype)
                if device.type == "cuda" and dtype != torch.float32 else contextlib.nullcontext())
    loss_sum, n_tok, t0 = 0.0, 0, time.time()
    with torch.no_grad():
        for batch in micro_batches(loader, args.batch):
            idx = batch.to(device, non_blocking=True)
            with make_acx():
                _, loss = model(idx[:, :-1], targets=idx[:, 1:])
            b, s = idx.shape[0], idx.shape[1] - 1
            loss_sum += float(loss) * b * s
            n_tok += b * s
    dt = time.time() - t0
    val = loss_sum / max(1, n_tok)
    name = os.path.basename(args.ckpt)
    print(f"{name}: val loss {val:.4f} | ppl {math.exp(min(20, val)):.3f} "
          f"| {n_tok:,} tok | {n_tok/dt/1e3:.1f}k tok/s (dtype={str(dtype).split('.')[-1]})")


if __name__ == "__main__":
    main()
