"""Stream-prepare the pretraining corpus into uint16 token shards + manifest.

  python -m neuron.prepare_data --spec configs/data_spec.yaml --out data

Spec (configs/data_spec.yaml):
    tokenizer: tokenizer/tokenizer.json
    val_interval: 997            # ~1/997 of docs (crc-hash) -> val
    doc_min_chars: 200
    doc_max_chars: 100_000
    doc_max_tokens: 12000
    shard_tokens: 16_777_216     # 16M tokens = 32MB per .bin
    buffer_tokens: 16_777_216    # doc-shuffle buffer before writing
    seq_len: 1024
    sources:
      - id: wikipedia_english
        kind: hf_streaming
        dataset: {path: oriental-lab/wikipedia-english}
        split: train
        text_field: text
        max_tokens: 0

Files: data/{train,val}_<source>_%06d.bin (uint16 little-endian)
Manifest: data/manifest.json (scans actual file sizes; authoritative)
Resume: data/progress.json (per-tokenizer shard counters; resumable mid-run)
"""
from __future__ import annotations

import argparse
import itertools as it
import json
import os
import sys
import time
import zlib
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
import yaml

from .specials import EOD


# --------------------------------------------------------------------------- #
# source readers
# --------------------------------------------------------------------------- #
def iter_examples(source: dict, seed: int, src_index: int = 0) -> Iterator[str]:
    kind = source.get("kind", "hf_streaming")
    if kind == "hf_streaming":
        from datasets import load_dataset

        dcfg = dict(source["dataset"])
        path = dcfg.pop("path")
        name = dcfg.pop("name", None)
        split = source.get("split", "train")
        text_field = source.get("text_field", "text")
        ds = load_dataset(path, name, split=split, streaming=True, **dcfg)
        if not source.get("stream_shuffle", True):
            pass
        else:
            try:
                ds = ds.shuffle(buffer_size=8192, seed=seed)  # stream-order shuffle
            except Exception:  # noqa: BLE001
                pass
        for row in ds:
            v = row.get(text_field)
            if v is None:
                continue
            if isinstance(v, list):
                for t in v:
                    if isinstance(t, str) and t:
                        yield t
            elif isinstance(v, str):
                yield v
    elif kind == "jsonl":
        field = source.get("text_field", "text")
        with open(source["path"], encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                v = row.get(field)
                if isinstance(v, str) and v:
                    yield v
    elif kind == "txt":
        sep = source.get("sep", "\n\n\n")
        with open(source["path"], encoding="utf-8", errors="ignore") as f:
            for chunk in f.read().split(sep):
                if chunk.strip():
                    yield chunk
    else:
        raise ValueError(f"unknown source kind {kind!r}")


def _val_flag(text: str, interval: int) -> bool:
    # stable across runs: crc32 of a slice of the doc, not doc order
    return zlib.crc32(text[:512].encode("utf-8", "ignore")) % interval == 0


# --------------------------------------------------------------------------- #
# writer
# --------------------------------------------------------------------------- #
class ShardWriter:
    """Buffer docs -> shuffle -> pack into fixed-size uint16 shard files."""

    def __init__(self, out_dir: Path, source_id: str, split: str, shard_tokens: int,
                 buffer_tokens: int, rng: np.random.Generator,
                 start_shard: int = 0, start_tokens: int = 0,
                 on_shard_done: Optional[Callable[[], None]] = None):
        self.out_dir = out_dir
        self.source_id = source_id
        self.split = split
        self.shard_tokens = int(shard_tokens)
        self.buffer_tokens = int(buffer_tokens)
        self.rng = rng
        self.shard_idx = int(start_shard)
        self.total = int(start_tokens)
        self.pending: list[np.ndarray] = []
        self.pending_tokens = 0
        self.on_shard_done = on_shard_done
        self._fp = None
        self._fp_tokens = 0
        self._open()

    # ---- file plumbing
    def _path(self, i: int) -> Path:
        return self.out_dir / f"{self.split}_{self.source_id}_{i:06d}.bin"

    def _open(self) -> None:
        self._close_fp()
        self._fp = open(self._path(self.shard_idx), "wb")
        self._fp_tokens = 0

    def _close_fp(self) -> None:
        if self._fp is not None and not self._fp.closed:
            self._fp.close()
        self._fp = None

    # ---- doc flow
    def add_doc(self, tokens: np.ndarray) -> None:
        if tokens.size == 0:
            return
        self.pending.append(tokens)
        self.pending_tokens += tokens.size
        if self.pending_tokens >= self.buffer_tokens:
            self.flush_buffer()

    def flush_buffer(self) -> None:
        self.rng.shuffle(self.pending)
        while self.pending:
            self._emit(self.pending.pop(0))

    def _emit(self, doc: np.ndarray) -> None:
        room = self.shard_tokens - self._fp_tokens
        if room <= 0:
            self._finish_shard()
            self._open()
            room = self.shard_tokens
        if doc.size <= room:
            doc.astype("<u2", copy=False).tofile(self._fp)
            self._fp_tokens += int(doc.size)
            self.total += int(doc.size)
        else:
            head, tail = doc[:room], doc[room:]
            head.astype("<u2", copy=False).tofile(self._fp)
            self._fp_tokens += int(head.size)
            self.total += int(head.size)
            self._finish_shard()
            self._open()
            self._emit(tail)

    def _finish_shard(self) -> None:
        if self._fp is None or self._fp_tokens == 0:
            if self._fp is not None:  # empty shard file -> delete
                self._fp.close()
                self._fp = None
                self._path(self.shard_idx).unlink(missing_ok=True)
            return
        self._fp.flush()
        self._fp.close()
        self._fp = None
        self.shard_idx += 1
        self._fp_tokens = 0
        if self.on_shard_done:
            self.on_shard_done()

    def close(self) -> None:
        self.flush_buffer()
        self._finish_shard()
        self._close_fp()


# --------------------------------------------------------------------------- #
# PROGRESS
# --------------------------------------------------------------------------- #
class Progress:
    """persistent.json — counters per (source, split), updated after each shard."""

    def __init__(self, path: Path):
        self.path = path
        self.state: dict = {}
        if path.exists():
            try:
                self.state = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                self.state = {}

    def get(self, sid: str, split: str) -> dict:
        return self.state.setdefault(sid, {}).setdefault(
            split, {"docs": 0, "tokens": 0, "shards": 0}
        )

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state), encoding="utf-8")
        os.replace(tmp, self.path)


def clean_stale_shards(out_dir: Path, prefix: str) -> int:
    removed = 0
    for p in sorted(out_dir.glob(prefix + "-*.bin")):
        p.unlink()
        removed += 1
    for p in out_dir.glob(prefix + "-*.bin.tmp"):
        p.unlink()
        removed += 1
    return removed


# --------------------------------------------------------------------------- #
# tokenize + write (per source)
# --------------------------------------------------------------------------- #
def run_source(tokenizer, source: dict, sid: str, out_dir: Path, opts: dict,
               prog: Progress, tokenizers_pass: int, seed: int) -> dict:
    st_train = prog.get(sid, "train")
    st_val = prog.get(sid, "val")
    interval = int(opts["val_interval"])
    min_chars = int(opts["doc_min_chars"])
    max_chars = int(opts["doc_max_chars"])
    max_tok = int(opts["doc_max_tokens"])
    target = int(source.get("max_tokens", 0))
    shard_tok = int(opts["shard_tokens"])
    buffer_tok = int(opts["buffer_tokens"])
    eod = tokenizer.token_to_id(EOD)
    if eod is None:
        eod = tokenizer.token_to_id("</s>")  # Llama 2 EOS
    bos = tokenizer.token_to_id("<s>")  # Llama 2 BOS; absent for the legacy BPE
    if eod is None:
        sys.exit(f"end-of-document token {EOD!r} or Llama EOS '</s>' missing")

    w_train = ShardWriter(out_dir, sid, "train", shard_tok, buffer_tok,
                          np.random.default_rng([seed, tokenizers_pass, 1]),
                          start_shard=st_train["shards"], start_tokens=st_train["tokens"])
    w_val = ShardWriter(out_dir, sid, "val", shard_tok, buffer_tok,
                        np.random.default_rng([seed, tokenizers_pass, 2]),
                        start_shard=st_val["shards"], start_tokens=st_val["tokens"])
    # Progress advances only at a commit boundary where both split buffers
    # have been flushed. This prevents a restart from skipping buffered docs.

    doc_skip = int(st_train["docs"])
    batch: list[str] = []
    batch_chars = 0
    docs_seen = 0
    t0 = time.time()
    beat_docs = 0

    def flush_batch() -> None:
        nonlocal batch, batch_chars
        if not batch:
            return
        encs = tokenizer.encode_batch(batch, add_special_tokens=False)
        for e, text in zip(encs, batch):
            content_ids = e.ids[:max_tok] if max_tok > 0 else e.ids
            ids = ([bos] if bos is not None else []) + list(content_ids) + [eod]
            arr = np.asarray(ids, dtype=np.uint16)
            if _val_flag(text, interval):
                w_val.add_doc(arr)
            else:
                w_train.add_doc(arr)
        batch = []
        batch_chars = 0

    def commit_progress(final: bool = False) -> None:
        flush_batch()
        train_first, val_first = w_train.shard_idx, w_val.shard_idx
        w_train.flush_buffer()
        w_val.flush_buffer()
        w_train._finish_shard()
        w_val._finish_shard()
        st_train["docs"] = doc_skip + docs_seen
        st_train["tokens"] = w_train.total
        st_train["shards"] = w_train.shard_idx
        st_val["tokens"] = w_val.total
        st_val["shards"] = w_val.shard_idx
        prog.save()
        try:
            from .hub import upload_prepared_files
            closed = [str(w_train._path(i)) for i in range(train_first, w_train.shard_idx)]
            closed += [str(w_val._path(i)) for i in range(val_first, w_val.shard_idx)]
            upload_prepared_files(str(out_dir), closed)
        except Exception as e:  # noqa: BLE001
            print(f"[hub][warn] data shard upload failed; local prep can resume: {e}", flush=True)
        if not final:
            w_train._open()
            w_val._open()

    stream = iter_examples(source, seed=seed + tokenizers_pass, src_index=tokenizers_pass)
    if doc_skip:
        stream = it.islice(stream, doc_skip, None)

    for text in stream:
        if target > 0 and w_train.total >= target:
            break
        docs_seen += 1
        if len(text) < min_chars:
            continue
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars]
        batch.append(text)
        batch_chars += len(text)
        if len(batch) >= 1024 or batch_chars >= 4_000_000:
            flush_batch()
        if (docs_seen - beat_docs) >= 20_000:
            beat_docs = docs_seen
            dt = max(1e-9, time.time() - t0)
            speed = (w_train.total + w_val.total - st_train["tokens"]) / dt
            eta_min = (max(0, target - w_train.total) / max(1.0, speed) / 60.0
                       if target > 0 else float("inf"))
            print(f"[{sid}] docs={doc_skip + docs_seen:,} "
                  f"tok={w_train.total:,}/{'all' if target <= 0 else f'{target:,}'} "
                  f"({speed/1e3:.0f}k tok/s, "
                  f"eta {'unknown' if target <= 0 else f'{eta_min:.0f}m'})", flush=True)
            commit_progress()

    commit_progress(final=True)

    docs_processed = doc_skip + docs_seen
    st_train["docs"] = docs_processed
    st_train["tokens"] = w_train.total
    st_train["shards"] = w_train.shard_idx
    st_val["tokens"] = w_val.total
    st_val["shards"] = w_val.shard_idx
    prog.save()
    return {"docs": docs_processed, "train_tokens": w_train.total, "val_tokens": w_val.total}


# --------------------------------------------------------------------------- #
# manifest builder (scans real files; authoritative sizes)
# --------------------------------------------------------------------------- #
def build_manifest(out_dir: Path, tok_path: str, vocab_size: int, sources: list[dict],
                   seq_len: int, eod_token_id: int, bos_token_id: int | None) -> dict:
    def entries(split: str) -> list[dict]:
        es = []
        for p in sorted(out_dir.glob(f"{split}_*.bin")):
            sid = p.name[len(split) + 1: p.name.rfind("_")]
            w = next((s.get("weight", 1.0) for s in sources if s.get("id") == sid), 1.0)
            es.append({"path": p.name, "tokens": p.stat().st_size // 2,
                       "shard_size": p.stat().st_size // 2, "source": sid, "weight": w})
        return es

    tr, va = entries("train"), entries("val")
    return {
        "version": 4,
        "vocab_size": vocab_size,
        "eod_token_id": int(eod_token_id),
        "bos_token_id": int(bos_token_id) if bos_token_id is not None else None,
        "tokenizer": os.path.relpath(tok_path, out_dir).replace("\\", "/"),
        "seq_len": int(seq_len),
        "splits": {"train": tr, "val": va},
        "totals": {"train": sum(e["tokens"] for e in tr), "val": sum(e["tokens"] for e in va)},
        "sources": sources,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="configs/data_spec.yaml")
    ap.add_argument("--out", default="data")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = tokenizer's internal parallelism (recommended)")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    from tokenizers import Tokenizer

    with open(args.spec, encoding="utf-8") as f:
        spec = yaml.safe_load(f) or {}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_path = args.tokenizer or spec.get("tokenizer", "tokenizer/tokenizer.json")
    if not os.path.exists(tok_path):
        sys.exit(f"tokenizer missing: {tok_path} — run `python -m neuron.train_tokenizer` first")
    tokenizer = Tokenizer.from_file(tok_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"[prep] tokenizer vocab={vocab_size}", flush=True)

    defaults = {
        "val_interval": 997,
        "doc_min_chars": 200,
        "doc_max_chars": 100_000,
        "doc_max_tokens": 12000,
        "shard_tokens": 16_777_216,
        "buffer_tokens": 16_777_216,
        "seq_len": 1024,
    }
    opts = {**defaults, **{k: v for k, v in spec.items() if k not in ("sources",)}}

    prog = Progress(out_dir / "progress.json")
    sources = spec.get("sources", [])
    had_progress = bool(prog.state)
    if not had_progress:
        removed = 0
        for s in sources:
            sid = s.get("id", "src")
            for split in ("train", "val"):
                for p in out_dir.glob(f"{split}_{sid}_*.bin"):
                    p.unlink()
                    removed += 1
                removed += clean_stale_shards(out_dir, f"{split}_{sid}")
        if removed:
            print(f"[prep] clean start: removed {removed} stale shard files")

    try:
        from datasets import disable_progress_bars

        disable_progress_bars()
    except Exception:  # noqa: BLE001
        pass

    totals = {"train": 0, "val": 0}
    for k, source in enumerate(sources):
        sid = source.get("id", f"src{k}")
        print(f"[prep] source {sid} (kind={source.get('kind')})", flush=True)
        info = run_source(tokenizer, source, sid, out_dir, opts, prog, k, 1337)
        totals["train"] += info["train_tokens"]
        totals["val"] += info["val_tokens"]
        print(f"[prep] {sid}: {info['train_tokens']:,} train tok "
              f"| {info['val_tokens']:,} val tok", flush=True)

    manifest = build_manifest(out_dir, tok_path, vocab_size, sources, opts["seq_len"],
                              tokenizer.token_to_id(EOD) if tokenizer.token_to_id(EOD) is not None
                              else tokenizer.token_to_id("</s>"),
                              tokenizer.token_to_id("<s>"))
    path = out_dir / "manifest.json"
    manifest_tmp = path.with_suffix(".tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    os.replace(manifest_tmp, path)
    print(f"[prep] manifest: train {manifest['totals']['train']:,} tok "
          f"| val {manifest['totals']['val']:,} tok -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
