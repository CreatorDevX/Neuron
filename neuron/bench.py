"""Throughput / memory benchmark for the target GPU.

  python -m neuron.bench --config configs/neuron-100m-t4.yaml

  --find-batch-size   tries micro_batch 8,16,24,32,48,64 and reports peak mem
  --steps N           measured steps (default 24; the first few are warmup)
  --grad-accum N      accumulation frozen for the bench (1 by default)

Uses synthetic tokens (no disk I/O) so the number reflects the train loop's
compute ceiling. Reports tok/s, MFU (vs 65 TFLOPS fp16 peak on T4), peak alloc.
"""
from __future__ import annotations

import argparse
import gc
import time

import torch

from .config import (
    ModelConfig,
    TrainConfig,
    approx_flops_per_token,
    build_configs,
    peak_flops,
)
from .model import GPT, configure_attention_flags
from .optim import build_optimizer, make_scaler

SIZES = [8, 16, 24, 32, 48, 64]


def bench_once(mcfg: ModelConfig, tcfg: TrainConfig, micro: int, accum: int, steps: int,
               device: torch.device, dtype: torch.dtype, compiled: bool):
    torch.cuda.empty_cache()
    gc.collect()
    model = GPT(mcfg).to(device)
    model.train()
    if compiled:
        model = torch.compile(model, mode=tcfg.compile_mode, dynamic=False)
    opt = build_optimizer(model, tcfg, device.type)
    scaler = make_scaler(tcfg, device.type, dtype)
    params = [p for p in model.parameters() if p.requires_grad]

    gen = torch.Generator(device="cpu").manual_seed(1337)
    x_pool = torch.randint(0, mcfg.vocab_size, (micro * accum, mcfg.seq_len), generator=gen)
    y_pool = x_pool.clone()  # same tokens: CE value is garbage; speed is what we measure

    quants = peak_flops()
    torch.cuda.reset_peak_memory_stats()
    try:
        for i in range(steps):
            if i == steps // 2:  # measure on the second half only
                torch.cuda.synchronize()
                t0 = time.time()
            opt.zero_grad(set_to_none=True)
            for a in range(accum):
                x = x_pool[a * micro:(a + 1) * micro].to(device, non_blocking=True)
                y = y_pool[a * micro:(a + 1) * micro].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype != torch.float32):
                    _, loss = model(x, targets=y)
                (scaler.scale(loss / accum) if scaler else loss / accum).backward()
            if scaler:
                scaler.unscale_(opt)
            _ = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
        torch.cuda.synchronize()
        dt = time.time() - t0
    except torch.cuda.OutOfMemoryError:
        del model, opt, scaler, params
        return None
    peak = torch.cuda.max_memory_allocated() / 2**30
    tokens = (steps - steps // 2) * micro * accum * mcfg.seq_len
    tok_s = tokens / dt
    mfu = tok_s * approx_flops_per_token(mcfg) / quants
    del model, opt, scaler, params, x_pool, y_pool
    torch.cuda.empty_cache()
    gc.collect()
    return {"micro": micro, "accum": accum, "tok_s": tok_s, "mfu": mfu, "peak_gb": peak,
            "step_s": dt / max(1, steps - steps // 2)}


def main() -> None:
    import time  # noqa: F401 (imported here to keep the module import cheap)

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/neuron-100m-t4.yaml")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--find-batch-size", action="store_true")
    ap.add_argument("--spec", type=int, nargs="*", default=None,
                    help="explicit micro-batch sizes; default 8,16,24,32,48,64")
    args = ap.parse_args()

    mcfg, tcfg = build_configs(args.config)
    configure_attention_flags(mcfg.attention)
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"bench on {name} (fp16 peak assumed {peak_flops()/1e12:.1f} TFLOPS), dtype={dtype}")

    if args.find_batch_size:
        for micro in (args.spec or SIZES):
            local = f"micro={micro}, accum={args.grad_accum}"
            try:
                r = bench_once(mcfg, tcfg, micro, args.grad_accum, args.steps, device, dtype,
                               args.compile)
            except torch.cuda.OutOfMemoryError:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                r = None
            if r is None or r is False:
                print(f"{local}: OOM")
                break
            print(f"{local}: {r['tok_s']:.0f} tok/s | MFU {r['mfu']*100:.1f}% | "
                  f"peak {r['peak_gb']:.2f} GB | {r['step_s']*1000:.0f} ms/step")
        return

    r = bench_once(mcfg, tcfg, tcfg.micro_batch, args.grad_accum, args.steps, device, dtype,
                   args.compile)
    if r is None:
        print("OOM: lower micro_batch")
        raise SystemExit(1)
    print(f"micro={r['micro']} accum={r['accum']}: {r['tok_s']:.0f} tok/s | MFU {r['mfu']*100:.1f}% "
          f"| peak {r['peak_gb']:.2f} GB | {r['step_s']*1000:.0f} ms/step")


if __name__ == "__main__":
    main()
