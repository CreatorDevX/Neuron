"""Training entry point.

Single GPU:
  python -m neuron.train --config configs/neuron-100m-t4.yaml

Notebook multi-GPU (Kaggle/Jupyter):
  from neuron.train import notebook_train
  notebook_train(num_processes=2)

CLI overrides: --set model.d_model=640 --set train.lr=6e-4 --set train.compile=false

Design notes:
  * fp16 autocast + GradScaler (Turing has no bf16; scaler absorbs inf/nan).
  * gradient accumulation; DDP `no_sync` on all but the last micro-batch.
  * WSD schedule; LR is a pure function of (step, total, warmup) -> resume-safe.
  * checkpoints on wall-clock AND step boundaries; auto-resume from latest.json.
  * SIGINT/SIGTERM are graceful: finish current step, checkpoint, exit.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
import subprocess
import signal
import sys
import time
from pathlib import Path
from typing import Iterator, List

import torch
import yaml

from . import __version__
from .checkpoint import (
    build_run_identity,
    load_checkpoint,
    prune_checkpoints,
    read_latest,
    save_checkpoint,
    snapshot_config,
    write_latest,
)
from .config import (
    ModelConfig,
    TrainConfig,
    approx_flops_per_token,
    build_configs,
    count_params,
    days_for_tokens,
    non_embedding_params,
    peak_flops,
    total_steps,
    warmup_steps_,
    world_tokens_per_step,
)
from .data import TokenDataset, load_manifest, make_loader
from .model import GPT, configure_attention_flags, unwrap
from .optim import build_optimizer, lr_at, make_scaler
from .specials import EOD


# --------------------------------------------------------------------------- #
# small utils
# --------------------------------------------------------------------------- #
def human_dt(sec: float) -> str:
    sec = int(sec)
    if sec < 90:
        return f"{sec}s"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m2 = divmod(m, 60)
    if h < 24:
        return f"{h}h{m2:02d}m"
    d, h2 = divmod(h, 24)
    return f"{d}d{h2:02d}h"


class Stopper:
    """Graceful stop: finish current optimizer step, then checkpoint & exit."""

    def __init__(self):
        self.stop = False
        signal.signal(signal.SIGINT, self._int)
        try:
            signal.signal(signal.SIGTERM, self._term)
        except (ValueError, OSError):
            pass

    def _int(self, *_):
        if self.stop:
            raise KeyboardInterrupt
        self.stop = True
        print("\n[stop] graceful: finishing current step, then checkpointing...", flush=True)

    def _term(self, *_):
        self.stop = True
        print("\n[stop] SIGTERM: will checkpoint and exit.", flush=True)


def resolve_amp(device: torch.device, requested: str):
    """-> (dtype, use_amp, need_scaler)."""
    if requested == "float32":
        return torch.float32, False, False
    if requested == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16, True, True
        return torch.float16, True, True
    dt = getattr(torch, requested)
    return dt, device.type == "cuda", dt == torch.float16


def autocast_make(device: torch.device, dtype: torch.dtype):
    def make():
        if device.type != "cuda" or dtype == torch.float32:
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, dtype=dtype)
    return make


def micro_batches(loader, mb: int, limit: int | None = None) -> Iterator[torch.Tensor]:
    """Group loader rows (one per item) into (mb, S+1) stacks."""
    buf: List[torch.Tensor] = []
    n = 0
    for row in loader:
        buf.append(row)
        if len(buf) == mb:
            yield torch.stack(buf)
            buf = []
            n += 1
            if limit is not None and n >= limit:
                break


class CudaBatchPrefetcher:
    """Stage pinned CPU batches onto CUDA while the current batch computes."""

    def __init__(self, loader_iter, device: torch.device):
        self.loader_iter = loader_iter
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.next_batch = None
        self._host_refs = []
        self._preload()

    def _preload(self) -> None:
        self._release_completed_hosts()
        try:
            host_batch = next(self.loader_iter)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = host_batch.to(self.device, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.stream)
        # Keep pinned source memory alive until its DMA transfer has completed.
        self._host_refs.append((event, host_batch))

    def _release_completed_hosts(self) -> None:
        self._host_refs = [(event, batch) for event, batch in self._host_refs if not event.query()]

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        if self.next_batch is None:
            raise StopIteration
        current = torch.cuda.current_stream(self.device)
        current.wait_stream(self.stream)
        batch = self.next_batch
        batch.record_stream(current)
        self._preload()
        return batch


# --------------------------------------------------------------------------- #
# eval + sampling
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_val(model, val_loader, device, make_acx, restore_grad_ckpt: bool) -> float:
    loader_mb = getattr(val_loader, "_neuron_mb", 8)
    core = unwrap(model)
    was_training = model.training
    model.eval()
    core.set_gradient_checkpointing(False)
    total = torch.zeros((), device=device, dtype=torch.float32)
    n_tok = 0
    for batch in micro_batches(val_loader, loader_mb):
        idx = batch.to(device, non_blocking=True)
        with make_acx():
            _, loss = core(idx[:, :-1], targets=idx[:, 1:])
        b, s = idx.shape[0], idx.shape[1] - 1
        total.add_(loss.detach().float() * (b * s))
        n_tok += b * s
    if was_training:
        model.train()
    core.set_gradient_checkpointing(restore_grad_ckpt)
    return float((total / max(1, n_tok)).item())


def _mb_of(loader):
    # infer micro-batch from dataset attr stored at build time
    return getattr(loader, "_neuron_mb", 8)


@torch.no_grad()
def sample_texts(model, tokenizer, prompts, bos_id, eos_id, device, make_acx, step: int) -> list[dict]:
    core = unwrap(model)
    was_training = core.training
    core.eval()
    rows = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        idx = torch.tensor([[bos_id] + ids], dtype=torch.long, device=device)
        with make_acx():
            out = core.generate(idx, max_new_tokens=48, temperature=0.8, top_k=50,
                                eos_id=eos_id)
        gen = tokenizer.decode(out[0, idx.shape[1]:].tolist(), skip_special_tokens=True)
        print(f"[sample] step {step:,} | {prompt[:36]!r} -> {gen[:140].replace(chr(10), ' ')!r}", flush=True)
        rows.append({"step": step, "prompt": prompt, "generation": gen, "t": time.time()})
    if was_training:
        core.train()
    return rows


def append_samples(out_dir: str, rows: list[dict]) -> None:
    if not rows:
        return
    path = Path(out_dir) / "samples.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_tokenizer(path: str, data_dir: str):
    try:
        from tokenizers import Tokenizer

        candidates = [path, os.path.join(data_dir, path)]
        manifest_path = os.path.join(data_dir, "manifest.json")
        if os.path.isfile(manifest_path):
            manifest = load_manifest(data_dir)
            manifest_tokenizer = manifest.get("tokenizer")
            if manifest_tokenizer:
                candidates.append(manifest_tokenizer if os.path.isabs(manifest_tokenizer)
                                  else os.path.join(data_dir, manifest_tokenizer))
        for cand in dict.fromkeys(candidates):
            if cand and os.path.exists(cand):
                return Tokenizer.from_file(cand)
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    if not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_TOKEN"):
        os.environ["WANDB_API_KEY"] = os.environ["WANDB_TOKEN"]
    t_start = time.time()
    ap = argparse.ArgumentParser(prog="neuron.train")
    ap.add_argument("--config", default="configs/neuron-100m-t4.yaml")
    ap.add_argument("--spec", default="configs/data_spec.yaml",
                    help="HF data spec used to bootstrap tokenizer and shards when absent")
    ap.add_argument("--no-auto-prepare", action="store_true",
                    help="require an existing tokenizer and prepared data")
    ap.add_argument("--train-new-tokenizer", action="store_true",
                    help="train a new BPE tokenizer during data setup; requires a fresh data directory")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--resume", default="auto", help="auto | none | <ckpt.pt>")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--disable-wandb", action="store_true",
                    help="disable W&B even when an API key is present")
    args = ap.parse_args(argv)
    if args.train_new_tokenizer and args.no_auto_prepare:
        ap.error("--train-new-tokenizer requires automatic data setup; omit --no-auto-prepare")

    mcfg, tcfg = build_configs(args.config, overrides=args.overrides)
    if args.max_steps:
        tcfg.max_steps = args.max_steps
    if args.out_dir:
        tcfg.out_dir = args.out_dir
    if args.data_dir:
        tcfg.data_dir = args.data_dir

    # Bootstrap from a fresh Kaggle checkout. Rank zero prepares
    # shared files before the process group is initialized; other ranks wait.
    rank = int(os.environ.get("RANK", "0"))
    manifest_file = os.path.join(tcfg.data_dir, "manifest.json")
    if not args.no_auto_prepare:
        if rank == 0:
            try:
                from .hub import sync_data
                if not os.path.isfile(manifest_file):
                    sync_data(tcfg.data_dir, download=True)
            except Exception as e:  # noqa: BLE001
                print(f"[hub][info] data restore skipped: {e}", flush=True)
            with open(args.spec, "r", encoding="utf-8") as f:
                spec_cfg = yaml.safe_load(f) or {}
            stale_data = os.path.isfile(manifest_file) or os.path.isfile(os.path.join(tcfg.data_dir, "progress.json"))
            if args.train_new_tokenizer and stale_data:
                raise RuntimeError(
                    "--train-new-tokenizer requires a fresh data directory; set train.data_dir to a new path"
                )
            expected_tok_repo = None if args.train_new_tokenizer else spec_cfg.get("tokenizer_repo")
            tok_meta_path = Path(tcfg.tokenizer_path).with_name("tokenizer.meta.json")
            actual_tok_repo = None
            if tok_meta_path.exists():
                try:
                    actual_tok_repo = json.loads(tok_meta_path.read_text(encoding="utf-8")).get("repo")
                except Exception:
                    pass
            tokenizer_ready = (not args.train_new_tokenizer) and os.path.isfile(tcfg.tokenizer_path) and (
                not expected_tok_repo or actual_tok_repo == expected_tok_repo
            )
            if expected_tok_repo and not tokenizer_ready and stale_data:
                raise RuntimeError(
                    f"prepared data does not use {expected_tok_repo}; use a fresh data/tokenizer directory "
                    "before switching tokenizer IDs"
                )
            if not tokenizer_ready:
                tokenizer_cmd = [
                    sys.executable, "-m", "neuron.train_tokenizer", "--spec", args.spec,
                    "--out", os.path.dirname(tcfg.tokenizer_path) or "tokenizer",
                ]
                if args.train_new_tokenizer:
                    tokenizer_cmd.append("--train-new")
                subprocess.run(tokenizer_cmd, check=True)
            if not os.path.isfile(manifest_file):
                subprocess.run([
                    sys.executable, "-m", "neuron.prepare_data", "--spec", args.spec,
                    "--out", tcfg.data_dir, "--tokenizer", tcfg.tokenizer_path,
                ], check=True)
                try:
                    from .hub import sync_data
                    sync_data(tcfg.data_dir, download=False)
                except Exception as e:  # noqa: BLE001
                    print(f"[hub][warn] prepared data upload failed: {e}", flush=True)
        elif not os.path.isfile(manifest_file):
            # Prep can take hours; avoid holding a short-lived NCCL rendezvous
            # open while rank zero streams and tokenizes the corpus.
            deadline = time.time() + 12 * 60 * 60
            while not os.path.isfile(manifest_file) and time.time() < deadline:
                time.sleep(15)
            if not os.path.isfile(manifest_file):
                raise TimeoutError("rank 0 did not finish data preparation within 12 hours")

    # ------------------------------------------------ torch knobs
    torch.backends.cuda.matmul.allow_tf32 = tcfg.allow_tf32
    torch.backends.cudnn.allow_tf32 = tcfg.allow_tf32
    torch.backends.cudnn.benchmark = False
    configure_attention_flags(mcfg.attention)
    if tcfg.anomaly_debug:
        torch.autograd.set_detect_anomaly(True)

    # ------------------------------------------------ distributed
    world = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        import torch.distributed as dist

        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        world, rank = dist.get_world_size(), dist.get_rank()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0
    torch.manual_seed(tcfg.seed + rank)

    dtype, use_amp, need_scaler = resolve_amp(device, tcfg.dtype)
    if device.type == "cpu" and dtype != torch.float32:
        print("[warn] CPU-only: forcing float32", flush=True)
        dtype, use_amp, need_scaler = torch.float32, False, False
    make_acx = autocast_make(device, dtype)

    # ------------------------------------------------ data
    os.makedirs(tcfg.out_dir, exist_ok=True)
    manifest = load_manifest(tcfg.data_dir)
    if tcfg.train_on_all_data:
        tcfg.total_tokens = int(manifest.get("totals", {}).get("train", 0))
        if tcfg.total_tokens <= 0:
            raise ValueError("train_on_all_data requested but the manifest has no training tokens")
        if is_main:
            print(f"[data] training across all {tcfg.total_tokens:,} train tokens", flush=True)
    run_identity = build_run_identity(tcfg.data_dir, tcfg.tokenizer_path)
    m_vocab = int(manifest.get("vocab_size", 0))
    if m_vocab and m_vocab != mcfg.vocab_size:
        raise ValueError(f"manifest vocab_size={m_vocab} != model vocab_size={mcfg.vocab_size}")

    seq = mcfg.seq_len
    train_ds = TokenDataset(tcfg.data_dir, "train", seq_len=seq, shuffle=True, seed=tcfg.seed)
    val_ds = TokenDataset(tcfg.data_dir, tcfg.val_split, seq_len=seq, shuffle=False, seed=tcfg.seed)
    if is_main:
        print(
            f"data: train {train_ds.stream.total:,} tok "
            f"({len(train_ds.stream.shards)} shards) | "
            f"val {val_ds.stream.total:,} tok | seq={seq}",
            flush=True,
        )

    steps_total = total_steps(tcfg, world, mcfg)
    warmup = warmup_steps_(tcfg, world, mcfg)
    tps_step = world_tokens_per_step(tcfg, world, mcfg)

    val_loader = make_loader(
        val_ds, start=0, count=tcfg.eval_batches * tcfg.eval_batch,
        num_workers=0, prefetch=2, pin=False, persistent=False,
    )
    val_loader._neuron_mb = tcfg.eval_batch

    # ------------------------------------------------ model
    model = GPT(mcfg).to(device)
    model.set_gradient_checkpointing(tcfg.grad_checkpointing)
    if tcfg.compile and device.type == "cuda":
        try:
            model = torch.compile(model, mode=tcfg.compile_mode, dynamic=tcfg.compile_dynamic)
            if is_main:
                print(f"[compile] mode={tcfg.compile_mode}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] torch.compile unavailable ({e}); running eager", flush=True)
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(
            model,
            device_ids=[int(os.environ.get("LOCAL_RANK", "0"))],
            gradient_as_bucket_view=True,
            bucket_cap_mb=50,
            broadcast_buffers=False,
            static_graph=True,
        )

    if is_main:
        n_all = count_params(mcfg)  # analytic (matches model construction)
        fl_tok = approx_flops_per_token(mcfg)
        peak = peak_flops()
        etas = "/".join(f"{days_for_tokens(tcfg.total_tokens, peak * world * f / fl_tok):.1f}"
                        for f in (0.2, 0.3, 0.4))
        print("=" * 76)
        print(f"NEURON v{__version__} | {device.type} | "
              f"dtype={str(dtype).split('.')[-1]} | world={world}")
        print(f"params {n_all/1e6:.1f}M ({non_embedding_params(mcfg)/1e6:.1f}M non-emb) | "
              f"tok/step {tps_step:,} | steps {steps_total:,} | warmup {warmup:,}")
        print(f"ETA days if MFU=20/30/40%: {etas}")
        print("=" * 76, flush=True)

    # ------------------------------------------------ optim
    optimizer = build_optimizer(model, tcfg, device.type)
    scaler = make_scaler(tcfg, device.type, dtype) if need_scaler else None
    params = [p for p in unwrap(model).parameters() if p.requires_grad]

    start_step, tokens_seen, best_val = 0, 0, math.inf
    remote_resume = None
    if args.resume == "auto":
        pointer = read_latest(tcfg.out_dir)
        path = pointer[1] if pointer else None
        if path is None and is_main:
            try:
                from .hub import restore_latest
                remote_resume = restore_latest(tcfg.out_dir)
            except Exception as e:  # noqa: BLE001
                print(f"[hub][warn] could not restore remote checkpoint: {e}", flush=True)
        if world > 1:
            box = [remote_resume]
            torch.distributed.broadcast_object_list(box, src=0)
            remote_resume = box[0]
        if path is None:
            path = remote_resume
    elif args.resume == "none":
        path = None
    else:
        path = args.resume

    if path:
        state = load_checkpoint(path, model=model, optimizer=optimizer,
                                scaler=scaler, map_location="cpu")
        saved_model_cfg = ModelConfig(**state["model_cfg"])
        if dataclasses.asdict(saved_model_cfg) != dataclasses.asdict(mcfg):
            raise ValueError("checkpoint model configuration differs; model architecture changes require a new run")
        saved_identity = state.get("identity")
        if saved_identity is None:
            if is_main:
                print("[warn] checkpoint predates input identity metadata; data/tokenizer match is unverified", flush=True)
        elif saved_identity != run_identity:
            raise ValueError("checkpoint data/tokenizer identity differs from the current inputs; use --resume none for a new run")
        start_step = int(state["step"])
        tokens_seen = int(state.get("tokens_seen", start_step * tps_step))
        best_val = float(state.get("best_val") or math.inf)
        if tcfg.max_steps > 0:
            steps_total = tcfg.max_steps
        else:
            remaining_tokens = max(0, tcfg.total_tokens - tokens_seen)
            steps_total = start_step + math.ceil(remaining_tokens / tps_step)
        warmup = warmup_steps_(tcfg, world, mcfg)
        print(f"[resume] {path} @ step {start_step:,} ({tokens_seen:,} tok); "
              f"target step {steps_total:,}", flush=True)
    else:
        snapshot_config(tcfg.out_dir, dataclasses.asdict(mcfg), dataclasses.asdict(tcfg),
                        manifests=run_identity)

    # ------------------------------------------------ logging
    log_path = Path(tcfg.out_dir) / "log.jsonl"
    wandb = None
    wandb_enabled = not args.disable_wandb and bool(args.wandb or os.environ.get("WANDB_API_KEY"))
    if is_main and wandb_enabled and (os.environ.get("WANDB_PROJECT") or tcfg.wandb_project):
        try:
            import wandb as _w

            run_id_file = Path(tcfg.out_dir) / "wandb_run_id.txt"
            run_id = os.environ.get("WANDB_RUN_ID")
            if not run_id and run_id_file.exists():
                run_id = run_id_file.read_text(encoding="utf-8").strip() or None
            if not run_id:
                run_id = _w.util.generate_id()
            run_id_file.write_text(run_id, encoding="utf-8")
            _w.init(project=os.environ.get("WANDB_PROJECT") or tcfg.wandb_project,
                    entity=os.environ.get("WANDB_ENTITY") or os.environ.get("WANDB_USERNAME") or None,
                    id=run_id, resume="allow",
                    config={"model": dataclasses.asdict(mcfg), "train": dataclasses.asdict(tcfg)})
            wandb = _w
        except Exception as e:  # noqa: BLE001
            print(f"[warn] wandb off: {e}", flush=True)

    def log(step, **kw):
        rec = {"t": round(time.time() - t_start, 1), "step": step, **kw}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        if wandb is not None:
            try:
                wandb.log({"train_step": step, **{k: v for k, v in kw.items()
                                                  if isinstance(v, (int, float))}})
            except Exception:
                pass

    tokenizer = get_tokenizer(tcfg.tokenizer_path, tcfg.data_dir)
    bos_id = (manifest.get("bos_token_id") if manifest.get("bos_token_id") is not None
              else manifest.get("eod_token_id", 0)) if tokenizer is not None else None
    eos_id = manifest.get("eod_token_id") if tokenizer is not None else None
    if tokenizer is None:
        print("[note] tokenizer file not found: periodic text samples disabled", flush=True)

    stopper = Stopper()

    def do_checkpoint(step_key: int, final: bool = False, rng_by_rank=None) -> str:
        ck_name = "final.pt" if final else f"ckpt-{step_key:09d}.pt"
        p = save_checkpoint(
            os.path.join(tcfg.out_dir, ck_name),
            model=model, optimizer=optimizer, scaler=scaler,
            step=step_key, tokens_seen=tokens_seen,
            model_cfg=dataclasses.asdict(mcfg),
            train_cfg=dataclasses.asdict(tcfg),
            best_val=(best_val if math.isfinite(best_val) else None),
            identity=run_identity,
            rng_by_rank=rng_by_rank,
        )
        write_latest(tcfg.out_dir, step_key, p)
        if not final:
            prune_checkpoints(tcfg.out_dir, tcfg.keep_last)
        if is_main:
            try:
                from .hub import upload_checkpoint
                upload_checkpoint(p, tcfg.out_dir)
            except Exception as e:  # noqa: BLE001
                print(f"[hub][warn] checkpoint upload failed (local checkpoint is safe): {e}", flush=True)
        return p

    # ------------------------------------------------ the loop
    accum = tcfg.grad_accum
    mb = tcfg.micro_batch
    steps_in_run = steps_total - start_step
    last_sample_step = -1

    def rank_stream() -> Iterator[int]:
        # block layout per step: [micro 0: rank0, rank1...][micro 1: ...]
        base = tokens_seen // seq
        for _step in range(start_step, steps_total):
            for micro in range(accum):
                first = base + (micro * world + rank) * mb
                yield from range(first, first + mb)
            base += accum * world * mb

    train_loader = make_loader(
        train_ds,
        batch_size=mb,
        num_workers=tcfg.num_workers,
        prefetch=tcfg.prefetch_factor,
        pin=tcfg.pin_memory and device.type == "cuda",
        persistent=tcfg.num_workers > 0,
        sampler=_RangeSampler(rank_stream(), steps_in_run * accum * mb),
    )
    train_iter = iter(train_loader)
    if device.type == "cuda":
        train_iter = CudaBatchPrefetcher(train_iter, device)

    window_t0, window_tokens = time.time(), 0
    ck_last_t = time.time()
    grad_norm = float("nan")
    step = start_step
    stopped_early = False

    try:
        while step < steps_total:
            if world > 1:
                stop_tensor = torch.tensor([int(stopper.stop) if is_main else 0], device=device)
                torch.distributed.broadcast(stop_tensor, src=0)
                stopper.stop = bool(stop_tensor.item())
            if stopper.stop:
                stopped_early = True
                break
            optimizer.zero_grad(set_to_none=True)
            # LR for step `step` is applied BEFORE this step's optimizer.step()
            lr_now = lr_at(step, tcfg, steps_total, warmup)
            for g in optimizer.param_groups:
                g["lr"] = lr_now
            t_step = time.time()
            loss_step = torch.zeros((), device=device, dtype=torch.float32)
            for micro in range(accum):
                # Transfer the overlapping input/target stream once, then take
                # views on-device instead of issuing two near-identical copies.
                batch = next(train_iter)
                if device.type != "cuda":
                    batch = batch.to(device)
                x, y = batch[:, :-1], batch[:, 1:]
                sync_ctx = model.no_sync() if world > 1 and micro < accum - 1 else contextlib.nullcontext()
                with sync_ctx:
                    with make_acx():
                        _, loss = model(x, targets=y, kv_chunk_size=tcfg.kv_chunk_size)
                    loss_micro = loss / accum
                    backward(loss_micro, scaler)
                # Keep loss aggregation on-device. Converting each micro-batch
                # loss to Python forces a CUDA sync and stalls the next batch.
                loss_step.add_(loss.detach().float() / accum)
                window_tokens += x.numel()

            next_step = step + 1
            clip_due = next_step % tcfg.grad_clip_every == 0
            grad_norm = step_with_clip(
                optimizer, scaler, params,
                tcfg.grad_clip if clip_due else None,
                measure=(next_step % tcfg.log_every == 0 or next_step == steps_total),
            )
            step += 1
            tokens_seen += tps_step

            now = time.time()
            log_due = (step % tcfg.log_every == 0 or step == steps_total)
            if log_due and world > 1:
                torch.distributed.all_reduce(loss_step, op=torch.distributed.ReduceOp.SUM)
                loss_step.div_(world)
            loss_value = float(loss_step.item()) if log_due else 0.0
            if is_main and log_due:
                # Reading this scalar synchronizes the GPU; do it only on log steps.
                grad_norm_value = float(grad_norm)
                dt = now - t_step
                elapsed = now - window_t0
                tok_s = window_tokens * world / max(1e-6, elapsed)
                mfu = tok_s * approx_flops_per_token(mcfg) / (peak_flops() * world)
                mem = (torch.cuda.max_memory_allocated() / 2**30) if device.type == "cuda" else 0.0
                eta = (steps_total - step) * max(1e-6, now - t_step)
                log(
                    step=step,
                    loss=round(loss_value, 4),
                    lr=lr_now,
                    gnorm=(round(grad_norm_value, 3) if math.isfinite(grad_norm_value) else -1.0),
                    tok_s=round(tok_s),
                    mfu=round(float(mfu), 4),
                    mem_gb=round(mem, 2),
                    scale=(scaler.get_scale() if scaler else 1.0),
                    step_s=round(dt, 3),
                    eta=human_dt(eta),
                    tokens_seen=tokens_seen,
                )
                print(
                    f"[train] step {step:,}/{steps_total:,} | loss {loss_value:.4f} "
                    f"| lr {lr_now:.2e} | {tok_s:,.0f} tok/s | MFU {mfu*100:.1f}% "
                    f"| grad {grad_norm_value:.3f} | VRAM {mem:.2f} GB "
                    f"| seen {tokens_seen:,} tokens | ETA {human_dt(eta)}",
                    flush=True,
                )
                window_tokens, window_t0 = 0, now

            eval_due = bool(tcfg.eval_every and step < steps_total and step % tcfg.eval_every == 0)
            if eval_due:
                if world > 1:
                    torch.distributed.barrier()
                if is_main:
                    t_ev = time.time()
                    val_loss = eval_val(model, val_loader, device, make_acx, tcfg.grad_checkpointing)
                    improved = val_loss < best_val
                    if improved:
                        best_val = val_loss
                    log(step=step, val_loss=round(val_loss, 4),
                        val_ppl=round(math.exp(min(20.0, val_loss)), 3),
                        best=improved, eval_s=round(time.time() - t_ev, 1))
                    print(f"[eval] step {step:,} | val loss {val_loss:.4f} "
                          f"| ppl {math.exp(min(20.0, val_loss)):.3f} "
                          f"| best={'yes' if improved else 'no'}", flush=True)
                if world > 1:
                    torch.distributed.barrier()
                if world > 1:
                    rng_by_rank = gather_rng_states()
                else:
                    rng_by_rank = [local_rng_state()]
                if is_main and improved:
                    save_checkpoint(
                        os.path.join(tcfg.out_dir, "best.pt"), model=model,
                        optimizer=optimizer, scaler=scaler, step=step,
                        tokens_seen=tokens_seen, model_cfg=dataclasses.asdict(mcfg),
                        train_cfg=dataclasses.asdict(tcfg), best_val=best_val,
                        identity=run_identity, rng_by_rank=rng_by_rank,
                    )

            due = (
                (tcfg.ckpt_every_steps > 0 and step % tcfg.ckpt_every_steps == 0)
                or (tcfg.ckpt_every_min > 0 and (time.time() - ck_last_t) >= tcfg.ckpt_every_min * 60.0)
            )
            if world > 1:
                due_tensor = torch.tensor([int(due) if is_main else 0], device=device)
                torch.distributed.broadcast(due_tensor, src=0)
                due = bool(due_tensor.item())

            sample_due = (
                (tcfg.sample_every > 0 and step % tcfg.sample_every == 0)
                or (tcfg.sample_on_eval and eval_due)
                or (tcfg.sample_on_checkpoint and due)
            )
            if sample_due and is_main and tokenizer is not None and bos_id is not None:
                try:
                    prompts = list(tcfg.sample_prompts) or ["The "]
                    samples = sample_texts(
                        model, tokenizer, prompts, bos_id, eos_id, device, make_acx, step
                    )
                    append_samples(tcfg.out_dir, samples)
                    last_sample_step = step
                    if samples and not due:
                        from .hub import upload_samples
                        upload_samples(tcfg.out_dir)
                except Exception as e:  # noqa: BLE001
                    print(f"[warn] sampling failed: {e}", flush=True)

            if due:
                rng_by_rank = gather_rng_states() if world > 1 else [local_rng_state()]
            if due and is_main:
                t_ck = time.time()
                p = do_checkpoint(step, rng_by_rank=rng_by_rank)
                ck_last_t = time.time()
                print(f"[ckpt] step {step:,} -> {os.path.basename(p)} "
                      f"({time.time() - t_ck:.1f}s)", flush=True)
    except KeyboardInterrupt:
        if not stopper.stop and is_main:
            print("\n[int] interrupt: checkpointing current state...", flush=True)
    finally:
        if stopped_early and is_main:
            pass  # checkpoint already saved in-loop on stopper.stop
        if wandb is not None:
            try:
                wandb.finish()
            except Exception:
                pass

    has_final = step > start_step
    if world > 1:
        final_tensor = torch.tensor([int(has_final) if is_main else 0], device=device)
        torch.distributed.broadcast(final_tensor, src=0)
        has_final = bool(final_tensor.item())
    if has_final:
        rng_by_rank = gather_rng_states() if world > 1 else [local_rng_state()]
    if is_main and has_final:
        if (tcfg.sample_on_checkpoint and last_sample_step != step
                and tokenizer is not None and bos_id is not None):
            try:
                prompts = list(tcfg.sample_prompts) or ["The "]
                samples = sample_texts(
                    model, tokenizer, prompts, bos_id, eos_id, device, make_acx, step
                )
                append_samples(tcfg.out_dir, samples)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] final sampling failed: {e}", flush=True)
        p = do_checkpoint(step, final=True, rng_by_rank=rng_by_rank)
        print(f"[final] step {step:,} -> {os.path.basename(p)} "
              f"| run wall {human_dt(time.time() - t_start)}", flush=True)
    if world > 1:
        import torch.distributed as dist

        dist.destroy_process_group()


def backward(loss: torch.Tensor, scaler) -> None:
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()


def local_rng_state() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() and torch.cuda.is_initialized() else None,
    }


def gather_rng_states() -> list:
    state = local_rng_state()
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, state)
    return gathered


def step_with_clip(optimizer, scaler, params, clip: float | None, *, measure: bool = False) -> torch.Tensor:
    """Unscale (scaler) -> optionally clip -> step; keep diagnostics on-device."""
    if scaler is not None:
        scaler.unscale_(optimizer)
    norm = None
    if clip is not None:
        norm = torch.nn.utils.clip_grad_norm_(params, clip)
    elif measure:
        # Compute a diagnostic norm on log steps without modifying gradients.
        norm = torch.nn.utils.clip_grad_norm_(params, float("inf"))
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    if norm is not None:
        return norm
    return torch.full((), float("nan"), device=params[0].device)


class _RangeSampler:
    def __init__(self, gen, length: int):
        self._gen, self._len = gen, int(length)

    def __iter__(self):
        return self._gen

    def __len__(self):
        return self._len


if __name__ == "__main__":
    main()


def _notebook_entry(argv):
    """Top-level worker required by accelerate.notebook_launcher spawn."""
    main(argv)


def notebook_train(profile="auto", num_processes=None, args=None):
    """Launch this trainer across notebook GPUs without torchrun.

    Call from a Kaggle/Jupyter cell before initializing CUDA in the kernel:
      from neuron.train import notebook_train
      notebook_train()
    """
    from accelerate import notebook_launcher

    profiles = {
        "colab": ("configs/colab-1xt4.yaml", 1),
        "kaggle": ("configs/kaggle-2xt4.yaml", 2),
    }
    if profile not in ("auto", *profiles):
        raise ValueError("profile must be 'auto', 'colab', or 'kaggle'")
    available = torch.cuda.device_count()
    if torch.cuda.is_initialized():
        raise RuntimeError("call notebook_train before creating CUDA tensors in the notebook kernel")
    if profile == "auto":
        profile = "kaggle" if available >= 2 else "colab"
    config, expected_gpus = profiles[profile]
    if num_processes is None:
        num_processes = expected_gpus
    if available < num_processes:
        raise RuntimeError(f"profile requests {num_processes} GPU(s), but only {available} are visible")
    argv = list(args or [])
    if "--config" not in argv:
        argv = ["--config", config, *argv]
    notebook_launcher(_notebook_entry, args=(argv,), num_processes=num_processes)
