"""Checkpointing: atomic writes, pruning, resume with RNG state.

Checkpoint layout (out_dir/):
  ckpt-{step:09d}.pt   full state (model + optim + scaler + RNG + cfg snapshot)
  best.pt              best val-loss checkpoint (full state, atomic)
  latest.json          {"step": int, "path": "..."} pointer for auto-resume
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path

import torch


def save_checkpoint(
    path: str,
    *,
    model,
    optimizer,
    scaler,
    step: int,
    tokens_seen: int,
    model_cfg: dict,
    train_cfg: dict,
    best_val: float | None = None,
    identity: dict | None = None,
    rng_by_rank: list | None = None,
) -> str:
    from .model import unwrap

    base = unwrap(model)
    state = {
        "format": 3,
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "model": base.state_dict(),
        "optim": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "model_cfg": model_cfg,
        "train_cfg": train_cfg,
        "best_val": best_val,
        "identity": identity,
        "rng_by_rank": rng_by_rank,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() and torch.cuda.is_initialized() else None,
        },
    }
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state, tmp, _use_new_zipfile_serialization=True)
    os.replace(tmp, path)  # atomic on POSIX & Windows
    return path


def write_latest(out_dir: str, step: int, path: str) -> None:
    latest = os.path.join(out_dir, "latest.json")
    tmp = latest + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"step": int(step), "path": os.path.relpath(path, out_dir)}, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, latest)


def build_run_identity(data_dir: str, tokenizer_path: str) -> dict:
    """Capture lightweight, stable input identity without hashing multi-GB shards."""
    manifest_path = os.path.join(data_dir, "manifest.json")
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = json.loads(manifest_bytes)
    shards = []
    for split, entries in manifest.get("splits", {}).items():
        for entry in entries:
            shard_path = entry["path"]
            if not os.path.isabs(shard_path):
                shard_path = os.path.join(data_dir, shard_path)
            stat = os.stat(shard_path)
            shards.append({"path": os.path.relpath(shard_path, data_dir),
                           "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                           "split": split})
    configured_tok = tokenizer_path
    if configured_tok and not os.path.isabs(configured_tok) and not os.path.isfile(configured_tok):
        configured_tok = os.path.join(data_dir, configured_tok)
    manifest_tok = manifest.get("tokenizer")
    if manifest_tok and not os.path.isabs(manifest_tok):
        manifest_tok = os.path.join(data_dir, manifest_tok)
    configured_hash = sha256_file(configured_tok) if configured_tok and os.path.isfile(configured_tok) else None
    manifest_hash = sha256_file(manifest_tok) if manifest_tok and os.path.isfile(manifest_tok) else None
    if configured_hash and manifest_hash and configured_hash != manifest_hash:
        raise ValueError("configured tokenizer differs from the tokenizer recorded in manifest.json")
    tok_hash = manifest_hash or configured_hash
    return {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "tokenizer_sha256": tok_hash,
        "shards": shards,
    }


def run_identity_extends(saved: dict, current: dict) -> bool:
    """Allow an append-only prepared stream to add shards between checkpoints."""
    if saved.get("tokenizer_sha256") != current.get("tokenizer_sha256"):
        return False
    current_shards = {
        (s.get("path"), s.get("size"), s.get("mtime_ns"), s.get("split"))
        for s in current.get("shards", [])
    }
    return all(
        (s.get("path"), s.get("size"), s.get("mtime_ns"), s.get("split")) in current_shards
        for s in saved.get("shards", [])
    )


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_latest(out_dir: str) -> tuple[int, str] | None:
    p = os.path.join(out_dir, "latest.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        target = d.get("path")
        if target and not os.path.isabs(target):
            target = os.path.join(out_dir, target)
        if target and os.path.exists(target):
            return int(d["step"]), target
    except Exception:
        pass
    # fall back to newest ckpt file
    cands = sorted(glob.glob(os.path.join(out_dir, "ckpt-*.pt")))
    if cands:
        latest = cands[-1]
        return int(os.path.basename(latest)[5:14]), latest
    return None


def load_checkpoint(
    path: str,
    *,
    model=None,
    optimizer=None,
    scaler=None,
    map_location: str = "cpu",
    strict: bool = True,
) -> dict:
    state = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:
        from .model import unwrap

        sd = state["model"]
        tgt = unwrap(model)
        missing, unexpected = tgt.load_state_dict(sd, strict=strict)
        if missing or unexpected:
            raise RuntimeError(f"ckpt mismatch: missing={missing} unexpected={unexpected}")
    if optimizer is not None and state.get("optim") is not None:
        optimizer.load_state_dict(state["optim"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    if model is not None and state.get("rng") is not None:
        rank = int(os.environ.get("RANK", "0"))
        rng = state["rng"]
        per_rank = state.get("rng_by_rank")
        if per_rank:
            rng = per_rank[min(rank, len(per_rank) - 1)]
        torch.set_rng_state(rng.get("torch").cpu())
        cuda_state = rng.get("cuda")
        if cuda_state is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
            try:
                torch.cuda.set_rng_state_all([s.cpu() for s in cuda_state])
            except Exception:
                pass
    return state


def prune_checkpoints(out_dir: str, keep_last: int, keep_step: int = 0) -> None:
    cands = sorted(
        glob.glob(os.path.join(out_dir, "ckpt-*.pt")),
        key=lambda p: int(os.path.basename(p)[5:14]),
    )
    for p in cands[:-keep_last] if keep_last is not None else cands:
        if keep_step and int(os.path.basename(p)[5:14]) == keep_step:
            continue
        try:
            os.remove(p)
        except OSError:
            pass


def snapshot_config(out_dir: str, model_cfg: dict, train_cfg: dict, manifests=None) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    target = os.path.join(out_dir, "config_used.json")
    tmp = target + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"model": model_cfg, "train": train_cfg, "inputs": manifests}, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target)
