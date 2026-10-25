"""Optional Hugging Face Storage Bucket checkpoint sync."""
from __future__ import annotations

import os
from pathlib import Path


def bucket_id() -> str | None:
    configured = os.environ.get("HF_BUCKET_ID")
    if configured:
        return configured
    name = os.environ.get("HF_BUCKET_NAME")
    owner = os.environ.get("HF_NAMESPACE") or os.environ.get("HF_USERNAME")
    return f"{owner}/{name}" if owner and name else None


def upload_checkpoint(path: str, out_dir: str) -> None:
    bucket = bucket_id()
    if not bucket:
        return
    from huggingface_hub import batch_bucket_files

    local = Path(path)
    remote = f"{os.environ.get('HF_BUCKET_PREFIX', 'neuron')}/checkpoints/{local.name}"
    run_id = Path(out_dir) / "wandb_run_id.txt"
    if run_id.exists():
        batch_bucket_files(bucket, add=[(str(run_id), f"{os.environ.get('HF_BUCKET_PREFIX', 'neuron')}/wandb_run_id.txt")])
    log = Path(out_dir) / "log.jsonl"
    if log.exists():
        batch_bucket_files(bucket, add=[(str(log), f"{os.environ.get('HF_BUCKET_PREFIX', 'neuron')}/log.jsonl")])
    batch_bucket_files(bucket, add=[(str(local), remote)])
    # Publish the resume pointer last, after the checkpoint and run metadata land.
    latest = Path(out_dir) / "latest.json"
    if latest.exists():
        batch_bucket_files(bucket, add=[(str(latest), f"{os.environ.get('HF_BUCKET_PREFIX', 'neuron')}/latest.json")])


def restore_latest(out_dir: str) -> str | None:
    bucket = bucket_id()
    if not bucket:
        return None
    out = Path(out_dir)
    latest = out / "latest.json"
    if latest.exists():
        try:
            import json
            current = json.loads(latest.read_text(encoding="utf-8"))
            current_path = Path(current["path"])
            if not current_path.is_absolute():
                current_path = out / current_path
            if current_path.exists():
                return None
        except Exception:
            pass
        latest.unlink(missing_ok=True)
    from huggingface_hub import download_bucket_files

    prefix = os.environ.get("HF_BUCKET_PREFIX", "neuron")
    remote_latest = f"{prefix}/latest.json"
    try:
        download_bucket_files(bucket, files=[(remote_latest, str(latest))])
        import json
        name = json.loads(latest.read_text(encoding="utf-8"))["path"]
        remote_ckpt = f"{prefix}/checkpoints/{Path(name).name}"
        local_ckpt = out / Path(name).name
        download_bucket_files(bucket, files=[(remote_ckpt, str(local_ckpt))])
        try:
            download_bucket_files(
                bucket,
                files=[(f"{prefix}/log.jsonl", str(out / "log.jsonl"))],
            )
        except Exception:
            pass
        try:
            download_bucket_files(
                bucket,
                files=[(f"{prefix}/wandb_run_id.txt", str(out / "wandb_run_id.txt"))],
            )
        except Exception:
            pass
        return str(local_ckpt)
    except Exception:
        latest.unlink(missing_ok=True)
        return None


def sync_data(data_dir: str, *, download: bool) -> None:
    bucket = bucket_id()
    if not bucket:
        return
    from huggingface_hub import sync_bucket

    prefix = os.environ.get("HF_BUCKET_PREFIX", "neuron")
    remote = f"hf://buckets/{bucket}/{prefix}/data"
    if download:
        sync_bucket(remote, data_dir)
        tok_dir = Path(data_dir).parent / "tokenizer"
        try:
            sync_bucket(f"hf://buckets/{bucket}/{prefix}/tokenizer", str(tok_dir))
        except Exception:
            # During preprocessing, data/progress may have been uploaded before
            # the tokenizer directory. The caller can recreate it from HF.
            pass
    else:
        sync_bucket(data_dir, remote)
        tok_dir = Path(data_dir).parent / "tokenizer"
        if tok_dir.exists():
            sync_bucket(str(tok_dir), f"hf://buckets/{bucket}/{prefix}/tokenizer")


def upload_prepared_files(data_dir: str, paths: list[str]) -> None:
    bucket = bucket_id()
    if not bucket:
        return
    from huggingface_hub import batch_bucket_files

    prefix = os.environ.get("HF_BUCKET_PREFIX", "neuron")
    add = [(p, f"{prefix}/data/{Path(p).name}") for p in paths if Path(p).is_file()]
    progress = Path(data_dir) / "progress.json"
    if add:
        batch_bucket_files(bucket, add=add)
    # Publish the resume cursor only after every shard at that cursor is stored.
    if progress.is_file():
        batch_bucket_files(bucket, add=[(str(progress), f"{prefix}/data/progress.json")])
