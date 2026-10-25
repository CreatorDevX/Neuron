"""Token shard storage + deterministic streaming sampler.

On-disk format: plain uint16 (little-endian) binary per shard, plus one
manifest.json:

    {
      "version": 1,
      "vocab_size": 32000,
      "tokenizer": "tokenizer/tokenizer.json",   # relative to manifest dir
      "seq_len": 1024,
      "splits": {
        "train": [{"path": "train_source_000000.bin", "tokens": 16777216, "source": "..."}],
        "val":   [{"path": "val_source_000000.bin",  "tokens": 33554432, "source": "..."}]
      },
      "totals": {"train": ..., "val": ...},
      "sources": [...]
    }

The sampler defines a "stream" = shards concatenated in a deterministic
(per-epoch, seeded) weighted-random order. Training reads contiguous blocks of
the stream, so an exact resume is just "tokens_seen": no RNG bookkeeping, no
duplicates within an epoch, no gaps. All I/O is tiny (32 bytes/fetched token via
memmap + int32 cast).
"""
from __future__ import annotations

import bisect
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class ShardInfo:
    path: str          # absolute path
    tokens: int
    shard_size: int    # tokens in the file (== tokens)
    source: str = ""
    weight: float = 1.0


def load_manifest(data_dir: str) -> dict:
    with open(os.path.join(data_dir, "manifest.json"), "r", encoding="utf-8") as f:
        return json.load(f)


class TokenStream:
    """Deterministic contiguous token stream over a set of shards.

    get(token_pos, n) returns n tokens (uint16) starting at token_pos of the
    epoch-permuted shard order; positions wrap into epoch+1 automatically.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        shuffle: bool = True,
        seed: int = 1337,
    ):
        manifest = load_manifest(data_dir)
        entries = manifest.get("splits", {}).get(split, [])
        if not entries:
            raise FileNotFoundError(f"manifest has no '{split}' shards in {data_dir}")
        self.data_dir = data_dir
        self.seq_len = int(manifest.get("seq_len", 1024))
        self.vocab_size = int(manifest.get("vocab_size", 0))
        self.shards: List[ShardInfo] = []
        for e in entries:
            path = e["path"] if os.path.isabs(e["path"]) else os.path.join(data_dir, e["path"])
            tokens = int(e["tokens"])
            shard_size = int(e.get("shard_size", tokens))
            if tokens <= 0 or shard_size < tokens:
                raise ValueError(f"invalid token counts in manifest entry {e!r}")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"manifest shard missing: {path}")
            actual_bytes = os.path.getsize(path)
            if actual_bytes % 2 or actual_bytes != shard_size * 2:
                raise ValueError(f"shard size mismatch for {path}: {actual_bytes} bytes, expected {shard_size * 2}")
            self.shards.append(
                ShardInfo(
                    path=path,
                    tokens=tokens,
                    shard_size=shard_size,
                    source=e.get("source", ""),
                    weight=float(e.get("weight", 1.0)),
                )
            )
        self.total = sum(s.tokens for s in self.shards)
        self.shuffle = shuffle
        self.seed = seed
        self._mm: Dict[str, np.memmap] = {}
        self._order_cache: Dict[int, tuple] = {}

    # ------------------------------------------------------------------ order
    def _order(self, epoch: int):
        """Weighted random permutation of shard order for this epoch (cached)."""
        if epoch in self._order_cache:
            return self._order_cache[epoch]
        n = len(self.shards)
        if not self.shuffle or n == 1:
            order = np.arange(n)
        else:
            rng = np.random.default_rng([self.seed, 0x5EED + epoch])
            w = np.array([max(1e-6, s.weight) for s in self.shards], dtype=np.float64)
            # Efraimidis–Spirakis: weighted sampling without replacement
            keys = rng.random(n) ** (1.0 / w)
            order = np.argsort(-keys, kind="stable")
        lengths = np.empty(n, dtype=np.int64)
        for j, idx in enumerate(order):
            lengths[j] = self.shards[int(idx)].tokens
        cum = np.cumsum(lengths)
        self._order_cache[epoch] = (order, cum)
        return self._order_cache[epoch]

    # --------------------------------------------------------------- reading
    def _mmap(self, path: str) -> np.memmap:
        mm = self._mm.get(path)
        if mm is None:
            mm = np.memmap(path, dtype="<u2", mode="r")
            self._mm[path] = mm  # benign racy emit under GIL; run imports kept simple
        return mm

    def get(self, token_pos: int, n: int) -> np.ndarray:
        token_pos = int(token_pos)
        if token_pos < 0:
            raise ValueError("token_pos must be >= 0")
        out = np.empty(n, dtype=np.uint16)
        filled = 0
        while filled < n:
            epoch = token_pos // self.total
            pos = token_pos % self.total
            order, cum = self._order(epoch % (1 << 32))
            i = bisect.bisect_right(cum, pos)
            i = min(i, len(order) - 1)
            before = int(cum[i - 1]) if i > 0 else 0
            local = pos - before
            shard = self.shards[int(order[i])]
            take = min(n - filled, shard.tokens - local)
            if take <= 0:
                # exactly at a shard boundary: advance position and retry
                token_pos += 1
                filled += 0
                continue
            out[filled: filled + take] = self._mmap(shard.path)[local: local + take]
            filled += take
            token_pos += take
        return out

    def close(self) -> None:
        for mm in list(self._mm.values()):
            try:
                mm._mmap.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        self._mm.clear()


class TokenDataset(Dataset):
    """Dataset over stream positions; item i = tokens [i*S, (i+1)*S+1)."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        seq_len: int = 1024,
        shuffle: bool = True,
        seed: int = 1337,
        start_index: int = 0,
    ):
        self.stream = TokenStream(data_dir, split=split, shuffle=shuffle, seed=seed)
        self.seq_len = int(seq_len)
        self.start_index = int(start_index)
        max_items = self.stream.total // self.seq_len
        self.end_index = max(self.start_index, max_items)

    def __len__(self) -> int:
        return max(0, self.end_index - self.start_index)

    def __getitem__(self, i: int) -> torch.Tensor:
        block = self.stream.get((self.start_index + i) * self.seq_len, self.seq_len + 1)
        return torch.from_numpy(block.astype(np.int32))

    def __getitems__(self, indices) -> torch.Tensor:
        """Fetch a consecutive loader batch with one stream read and one cast.

        PyTorch's batched map-style fetch calls this once per batch. Fall back
        to individual reads for non-consecutive/custom sampler indices.
        """
        if not indices:
            return torch.empty((0, self.seq_len + 1), dtype=torch.int32)
        absolute = [self.start_index + int(i) for i in indices]
        if all(b == a + 1 for a, b in zip(absolute, absolute[1:])):
            length = len(absolute) * self.seq_len + 1
            block = self.stream.get(absolute[0] * self.seq_len, length)
            windows = np.lib.stride_tricks.sliding_window_view(block, self.seq_len + 1)[
                ::self.seq_len
            ][:len(absolute)]
            return torch.from_numpy(windows.astype(np.int32, copy=True))
        rows = [self.stream.get(i * self.seq_len, self.seq_len + 1) for i in absolute]
        return torch.from_numpy(np.stack(rows).astype(np.int32, copy=False))

    def close(self) -> None:
        self.stream.close()


def _collate_token_rows(rows):
    """Accept either the pre-batched __getitems__ result or individual rows."""
    if torch.is_tensor(rows):
        return rows
    return torch.stack(rows)


def make_loader(
    ds: TokenDataset,
    start: Optional[int] = None,
    count: Optional[int] = None,
    *,
    batch_size: Optional[int] = None,
    num_workers: int = 0,
    prefetch: int = 8,
    pin: bool = True,
    persistent: bool = True,
    sampler=None,
) -> DataLoader:
    """Loader yielding pre-packed (seq_len+1) int32 blocks or microbatches.

    If `sampler` is given, it must yield absolute stream block indices (the
    dataset uses start_index=0 in that case; see neuron.train for the layout).
    Setting `batch_size` lets workers collate complete microbatches before they
    cross the worker queue, avoiding per-sequence queue and main-thread stack
    overhead in the training loop.
    """
    if sampler is None:
        start = ds.start_index if start is None else start
        count = len(ds) if count is None else count
        sampler = range(start, start + count)
    if num_workers > 0:
        return DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            prefetch_factor=max(2, int(prefetch)),
            pin_memory=pin,
            persistent_workers=persistent,
            collate_fn=_collate_token_rows,
        )
    return DataLoader(
        ds, batch_size=batch_size, sampler=sampler, num_workers=0,
        pin_memory=pin,
        collate_fn=_collate_token_rows,
    )


def micro_batch_index(step: int, micro: int, accum: int, world: int, rank: int, mb: int) -> int:
    """First stream index of the (step, micro) micro-batch on `rank` of `world`.

    Stream layout per optimizer step: `accum * world` blocks of `mb` sequences
    interleaved as [(step,micro=0): ranks 0..w-1], ..., [(step,micro=a-1): ...].
    Training reads disjoint contiguous regions on each rank, so an exact resume
    at `step` lands on the exact same tokens.
    """
    return ((step * accum + micro) * world + rank) * mb


def shard_bytes_and_files(data_dir: str) -> tuple:
    manifest = load_manifest(data_dir)
    tot = 0
    files = 0
    for split in ("train", "val"):
        for e in manifest.get("splits", {}).get(split, []):
            files += 1
            tot += int(e["tokens"])
    return tot, files


def check_vocab(data_dir: str, tokenizer) -> None:
    manifest = load_manifest(data_dir)
    v = int(manifest.get("vocab_size", 0))
    if v and hasattr(tokenizer, "get_vocab_size") and tokenizer.get_vocab_size() != v:
        raise ValueError(
            f"tokenizer vocab {tokenizer.get_vocab_size()} != manifest vocab {v} "
            "(wrong tokenizer/data pair)"
        )
