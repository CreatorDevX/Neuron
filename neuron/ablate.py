"""Kernel and model-option performance ablations with synthetic random tokens.

Examples (run on the target T4/2xT4 environment):
  python -m neuron.ablate --config configs/neuron-100m-t4.yaml --device cuda
  python -m neuron.ablate --config configs/neuron-block-sparse-t4.yaml --device cuda --sweep all
  python -m neuron.ablate --config configs/smoke-test.yaml --device cpu --sweep attention

Each sweep is one-factor-at-a-time from the supplied config, so results are
readable and the number of runs stays manageable. Output is JSONL for later
plotting/comparison. This benchmarks one process/GPU; launch one process per
GPU for a 2xT4 comparison, or use torchrun only after adding DDP explicitly.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import itertools
import json
import os
import time
from pathlib import Path

import torch

from .config import ModelConfig, TrainConfig, build_configs, peak_flops
from .model import GPT, configure_attention_flags


ATTENTION = ("auto", "efficient", "math", "flash", "flash_turing", "block_sparse", "sparse")


def _parse_csv(value: str, cast=str):
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def make_cases(base: ModelConfig, args) -> list[tuple[str, ModelConfig, str, dict]]:
    """Return (case name, config, optional skip reason) one-factor variants."""
    cases: list[tuple[str, ModelConfig, str, dict]] = []

    def add(label: str, *, runtime=None, **changes):
        values = dataclasses.asdict(base)
        values.update(changes)
        cfg = ModelConfig(**values)
        reason = ""
        try:
            cfg.validate()
        except Exception as exc:  # invalid sweep values should be reported, not abort the sweep
            reason = str(exc)
        cases.append((label, cfg, reason, runtime or {}))

    if args.sweep in ("all", "attention"):
        for backend in _parse_csv(args.attention):
            changes = {"attention": backend}
            if backend == "block_sparse":
                changes["local_window"] = min(base.local_window, base.seq_len)
            add(f"attention={backend}", **changes)
    if args.sweep in ("all", "sequence"):
        for length in _parse_csv(args.seq_lens, int):
            add(f"seq_len={length}", seq_len=length, local_window=min(base.local_window, length),
                loss_chunk=min(base.loss_chunk, length))
    if args.sweep in ("all", "batch"):
        for batch in _parse_csv(args.batches, int):
            add(f"micro_batch={batch}")
    if args.sweep in ("all", "gqa"):
        for heads in _parse_csv(args.kv_heads, int):
            add(f"n_kv_heads={heads}", n_kv_heads=heads)
    if args.sweep in ("all", "engram"):
        for enabled in (False, True):
            add(f"use_engram={str(enabled).lower()}", use_engram=enabled)
    if args.sweep in ("all", "gdn"):
        for enabled in (False, True):
            add(f"hybrid_gdn={str(enabled).lower()}", hybrid_gdn=enabled)
        for backend in _parse_csv(args.gdn_backends):
            add(f"gdn_backend={backend}", hybrid_gdn=True, gdn_backend=backend)
    if args.sweep in ("all", "loss"):
        for chunk in _parse_csv(args.loss_chunks, int):
            add(f"loss_chunk={chunk}", loss_chunk=chunk)
    if args.sweep in ("all", "block_sparse"):
        for block, local, top in itertools.product(
            _parse_csv(args.block_sizes, int), _parse_csv(args.local_windows, int),
            _parse_csv(args.top_blocks, int),
        ):
            add(f"block={block},local={local},top={top}", attention="block_sparse",
                sparse_block_size=block, local_window=local, sparse_top_blocks=top)
    if args.sweep in ("all", "topk"):
        for topk in _parse_csv(args.top_ks, int):
            add(f"sparse_top_k={topk}", attention="sparse", sparse_top_k=topk)
    if args.sweep in ("all", "kv_chunk"):
        for chunk in _parse_csv(args.kv_chunks, int):
            add(f"kv_chunk_size={chunk}", runtime={"kv_chunk": chunk})
    if args.sweep in ("all", "runtime"):
        for compiled, checkpointing in itertools.product((False, True), repeat=2):
            add(f"compile={compiled},grad_checkpointing={checkpointing}", runtime={
                "compile": compiled, "grad_checkpointing": checkpointing,
            })
    return cases


def run_train_case(name: str, cfg: ModelConfig, skip: str, args, device: torch.device,
                   dtype: torch.dtype, micro_batch: int, runtime: dict) -> dict:
    kv_chunk = runtime.get("kv_chunk", 0)
    compiled = runtime.get("compile", args.compile)
    checkpointing = runtime.get("grad_checkpointing", args.grad_checkpointing)
    row = {"case": name, "attention": cfg.attention, "seq_len": cfg.seq_len,
           "micro_batch": micro_batch, "dtype": str(dtype), "kv_chunk_size": kv_chunk}
    if skip:
        return {**row, "status": "skipped", "reason": skip}
    if cfg.attention == "flash_turing" and device.type != "cuda":
        return {**row, "status": "skipped", "reason": "flash_turing requires CUDA"}
    if kv_chunk and (compiled or checkpointing):
        return {**row, "status": "skipped", "reason": "kv_chunk requires eager mode without gradient checkpointing"}
    try:
        cfg.validate()
        configure_attention_flags(cfg.attention)
        torch.manual_seed(args.seed)
        model = GPT(cfg).to(device)
        model.train()
        model.set_gradient_checkpointing(checkpointing)
        if compiled:
            model = torch.compile(model, mode=args.compile_mode)
        x = torch.randint(cfg.vocab_size, (micro_batch, cfg.seq_len), device=device)
        y = torch.randint(cfg.vocab_size, x.shape, device=device)
        use_amp = device.type == "cuda" and dtype != torch.float32

        def step():
            model.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                _, loss = model(x, targets=y, kv_chunk_size=kv_chunk)
            loss.backward()
            return loss

        for _ in range(args.warmup):
            step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(args.iters):
            loss = step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        tokens = args.iters * micro_batch * cfg.seq_len
        peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
        row.update(status="ok", mode="train", seconds=elapsed, ms_per_step=1000 * elapsed / args.iters,
                   tokens_per_second=tokens / elapsed, peak_allocated_gb=peak,
                   loss=float(loss.detach().float().cpu()), params=sum(p.numel() for p in model.parameters()))
        del model, x, y
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return row
    except Exception as exc:  # record per-kernel failures and continue the matrix
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return {**row, "status": "error", "reason": f"{type(exc).__name__}: {exc}"}


@torch.no_grad()
def run_decode_case(cfg: ModelConfig, args, device: torch.device, dtype: torch.dtype,
                    micro_batch: int, bounded_replay: bool) -> dict:
    cache_used = not args.no_cache and cfg.attention != "sparse"
    name = f"decode/cache={cache_used}/bounded_replay={bounded_replay}"
    row = {"case": name, "mode": "decode", "attention": cfg.attention,
           "seq_len": cfg.seq_len, "micro_batch": micro_batch, "dtype": str(dtype)}
    if args.no_cache and bounded_replay:
        return {**row, "status": "skipped", "reason": "bounded replay requires KV caching"}
    if bounded_replay and cfg.attention != "block_sparse":
        return {**row, "status": "skipped", "reason": "bounded replay requires block_sparse attention"}
    try:
        configure_attention_flags(cfg.attention)
        model = GPT(cfg).to(device).eval()
        x = torch.randint(cfg.vocab_size, (micro_batch, cfg.seq_len), device=device)
        use_cache = cache_used
        autocast_enabled = device.type == "cuda" and dtype != torch.float32
        for _ in range(args.warmup):
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                if use_cache:
                    model.generate(x, args.decode_tokens, temperature=0, top_k=0,
                                   swa_bounded_replay=bounded_replay)
                else:
                    _decode_without_cache(model, x, args.decode_tokens)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        # Generate's normal path caches automatically; --no-cache forces the
        # reference full-recompute behavior by temporarily using legacy sparse.
        start = time.perf_counter()
        for _ in range(args.iters):
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                if use_cache:
                    model.generate(x, args.decode_tokens, temperature=0, top_k=0,
                                   swa_bounded_replay=bounded_replay)
                else:
                    _decode_without_cache(model, x, args.decode_tokens)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        token_count = args.iters * micro_batch * args.decode_tokens
        row.update(status="ok", seconds=elapsed, generated_tokens_per_second=token_count / elapsed,
                   ms_per_generated_token=1000 * elapsed / token_count,
                   peak_allocated_gb=(torch.cuda.max_memory_allocated(device) / 2**30
                                      if device.type == "cuda" else None))
        del model, x
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return row
    except Exception as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return {**row, "status": "error", "reason": f"{type(exc).__name__}: {exc}"}


@torch.no_grad()
def _decode_without_cache(model: GPT, prompt: torch.Tensor, count: int) -> None:
    tokens = prompt
    for _ in range(count):
        logits, _ = model(tokens, last_only=True)
        token = logits[:, -1].argmax(-1, keepdim=True)
        tokens = torch.cat((tokens, token), dim=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/neuron-100m-t4.yaml")
    ap.add_argument("--device", default="cuda", help="cuda, cuda:0, cuda:1, or cpu")
    ap.add_argument("--sweep", choices=("all", "attention", "sequence", "batch", "gqa", "engram",
                                         "loss", "block_sparse", "topk", "kv_chunk", "runtime", "gdn"), default="all")
    ap.add_argument("--attention", default=",".join(ATTENTION))
    ap.add_argument("--seq-lens", default="512,1024")
    ap.add_argument("--batches", default="1,2,4")
    ap.add_argument("--kv-heads", default="1,2,4,8,12")
    ap.add_argument("--loss-chunks", default="64,128,256,512")
    ap.add_argument("--block-sizes", default="64,128,256")
    ap.add_argument("--local-windows", default="64,128,256")
    ap.add_argument("--top-blocks", default="0,2,4,8")
    ap.add_argument("--top-ks", default="128,512,1024,2048")
    ap.add_argument("--kv-chunks", default="0,128,256")
    ap.add_argument("--gdn-backends", default="auto,torch,fla")
    ap.add_argument("--modes", default="train", help="comma-separated train,decode,all")
    ap.add_argument("--decode-tokens", type=int, default=64)
    ap.add_argument("--no-cache", action="store_true", help="measure slow full-recompute decode instead of KV-cache decode")
    ap.add_argument("--bounded-replay", action="store_true", help="include SWA bounded-replay decode")
    ap.add_argument("--micro-batch", type=int, default=0, help="override config batch for every run")
    ap.add_argument("--dtype", default="auto", help="auto or comma-separated fp16,bf16,fp32")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--compile-mode", default="default")
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--output", default="runs/ablation.jsonl")
    args = ap.parse_args()

    if args.iters < 1 or args.warmup < 0 or args.decode_tokens < 1:
        ap.error("iters/decode-tokens must be positive and warmup non-negative")
    mcfg, tcfg = build_configs(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        print(f"GPU: {torch.cuda.get_device_name(device)} | peak reference: {peak_flops()/1e12:.1f} TFLOP/s")
    if args.dtype == "auto":
        dtypes = [torch.float16 if device.type == "cuda" else torch.float32]
    else:
        names = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        try:
            dtypes = [names[n] for n in _parse_csv(args.dtype)]
        except KeyError as exc:
            ap.error(f"unknown dtype {exc.args[0]!r}; choose fp16,bf16,fp32")
    modes = set(_parse_csv(args.modes))
    if not modes or modes - {"train", "decode", "all"}:
        ap.error("--modes must contain train, decode, and/or all")
    if "all" in modes:
        modes = {"train", "decode"}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    base_batch = args.micro_batch or tcfg.micro_batch
    cases = make_cases(mcfg, args)
    print(f"Running {len(cases)} one-factor cases; results -> {output}")
    with output.open("w", encoding="utf-8") as f:
        for name, cfg, skip, runtime in cases:
            for dtype in dtypes:
                if "train" in modes:
                    batch = int(name.split("=", 1)[1]) if name.startswith("micro_batch=") else base_batch
                    row = run_train_case(name, cfg, skip, args, device, dtype, batch, runtime)
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    print(json.dumps(row, sort_keys=True), flush=True)
                if "decode" in modes and name.startswith("attention="):
                    for replay in ([False, True] if args.bounded_replay else [False]):
                        row = run_decode_case(cfg, args, device, dtype, base_batch, replay)
                        f.write(json.dumps(row) + "\n")
                        f.flush()
                        print(json.dumps(row, sort_keys=True), flush=True)
    print("Done")


if __name__ == "__main__":
    main()
