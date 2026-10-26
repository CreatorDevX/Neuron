"""Decoder-only transformer (Llama-style) tuned for a T4 budget.

Architecture:
  * pre-norm + RMSNorm (fp32 statistics), SwiGLU MLP, no biases
  * RoPE (NeoX half-split layout) computed in fp32
  * grouped-query attention (GQA 12q/4kv)
  * experimental chunked causal top-k attention
  * optional hashed n-gram memory with gated residual fusion
"""
from __future__ import annotations

import contextlib
import math
import warnings
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    _HAS_ATTENTION_CTX = True
except Exception:
    _HAS_ATTENTION_CTX = False


def attn_context(backend: str):
    if backend == "auto":
        return contextlib.nullcontext()
    if _HAS_ATTENTION_CTX:
        cmap = {"math": [SDPBackend.MATH], "efficient": [SDPBackend.EFFICIENT_ATTENTION],
                "flash": [SDPBackend.FLASH_ATTENTION]}
        if backend in cmap:
            return sdpa_kernel(cmap[backend])
        return contextlib.nullcontext()
    return contextlib.nullcontext()


def configure_attention_flags(backend: str) -> None:
    if backend in ("auto", "flash_turing", "block_sparse"):
        return
    try:
        import torch.backends.cuda as c
        c.enable_flash_sdp(backend == "flash")
        c.enable_mem_efficient_sdp(backend in ("efficient", "auto"))
        c.enable_math_sdp(True)
    except Exception:
        pass


class _FlashAttentionTuring(torch.autograd.Function):
    """Autograd bridge for ssiu/flash-attention-turing's SM75 extension.

    The upstream Python interface lives beside its extension rather than in
    the installed package, so keep this small adapter in Neuron. Inputs use
    the extension's [batch, sequence, heads, head_dim] layout and retain GQA
    without expanding K/V heads.
    """

    @staticmethod
    def forward(ctx, q, k, v):
        try:
            import flash_attn_turing
        except ImportError as exc:
            raise RuntimeError(
                "attention='flash_turing' requires the separately built "
                "flash-attn-turing extension (SM75 / T4)."
            ) from exc

        scale = q.shape[-1] ** -0.5
        out, lse = flash_attn_turing.fwd(q.contiguous(), k.contiguous(), v.contiguous(), scale, True)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.scale = scale
        ctx.extension = flash_attn_turing
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = ctx.extension.bwd(
            q.contiguous(), k.contiguous(), v.contiguous(), out.contiguous(),
            lse.contiguous(), dout.contiguous(), ctx.scale, True,
        )
        return dq, dk, dv


@dataclass
class KVCache:
    """Per-layer attention state plus token history for chunked train/decode.

    Keys/values retain their autograd graph. In training, keep a cache only
    within one forward/backward sequence; do not carry it across optimizer
    steps or unrelated examples.
    """

    layers: tuple
    token_ids: torch.Tensor
    start_position: int = 0

    @property
    def position(self) -> int:
        return self.token_ids.shape[1]

    def detach(self) -> "KVCache":
        return KVCache(
            tuple(
                GDNState(state.conv.detach(), state.recurrent.detach())
                if isinstance(state, GDNState)
                else (state[0].detach(), state[1].detach())
                for state in self.layers
            ),
            self.token_ids,
            self.start_position,
        )


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)
        orig_dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(orig_dtype)


def rope_cos_sin(seq_len: int, head_dim: int, theta: float, device, dtype=torch.float32):
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # Keep the rotation in the activation dtype. Computing the whole Q/K
    # tensor in fp32 creates large transient allocations and elementwise work
    # on T4; the small RoPE tables are still generated in fp32 for accuracy.
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos[None, None, :, :].to(dtype=x.dtype)
    s = sin[None, None, :, :].to(dtype=x.dtype)
    return torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)


def _make_sparse_block_mask(seq_len: int, block_size: int, local_window: int, top_blocks: int) -> torch.Tensor:
    """Fixed, causal landmark routing for the block-sparse attention path.

    Each query block gets a token-exact local window plus up to `top_blocks`
    evenly spaced older blocks. The local window is applied separately at
    token granularity; this matrix contains only whole remote blocks.
    """
    n_blocks = (seq_len + block_size - 1) // block_size
    mask = torch.zeros((n_blocks, n_blocks), dtype=torch.bool)
    for q_block in range(n_blocks):
        q_start = q_block * block_size
        local_start = max(0, q_start - local_window + 1)
        # A remote block is included only if every token in it precedes the
        # local interval. This avoids overlap and double-counting.
        eligible = local_start // block_size
        count = min(eligible, top_blocks)
        if count == 0:
            continue
        if count == eligible:
            chosen = list(range(eligible))
        elif count == 1:
            chosen = [0]
        else:
            chosen = sorted({round(i * (eligible - 1) / (count - 1)) for i in range(count)})
        if chosen:
            mask[q_block, chosen] = True
    return mask


def _block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int,
    local_window: int,
    query_offset: int = 0,
    key_offset: int = 0,
) -> torch.Tensor:
    """Reference block-sparse attention using PyTorch SDPA.

    Inputs use [batch, sequence, heads, head_dim]. Each query block attends to
    its routed remote blocks and an exact causal sliding window. The selected
    key set is compacted before SDPA so the implementation avoids constructing
    a dense sequence-by-sequence attention matrix.
    """
    _, q_len, q_heads, _ = q.shape
    k_len, kv_heads = k.shape[1], k.shape[2]
    if q_heads % kv_heads:
        raise ValueError("query head count must be divisible by KV head count")

    qh = q.transpose(1, 2)
    kh = k.transpose(1, 2)
    vh = v.transpose(1, 2)
    if q_heads != kv_heads:
        repeats = q_heads // kv_heads
        kh = kh.repeat_interleave(repeats, dim=1)
        vh = vh.repeat_interleave(repeats, dim=1)

    chunks = []
    key_end = key_offset + k_len
    q_start = 0
    while q_start < q_len:
        abs_q_start = query_offset + q_start
        q_end = min(q_start + block_size - (abs_q_start % block_size), q_len)
        abs_q_end = query_offset + q_end
        query_block = abs_q_start // block_size
        local_start = max(0, abs_q_start - local_window + 1)

        if query_block < block_mask.shape[0]:
            remote_blocks = block_mask[query_block].to(device=q.device, dtype=torch.bool).nonzero(as_tuple=False).flatten()
            remote_blocks = remote_blocks[remote_blocks * block_size + block_size <= local_start]
            remote_abs = (remote_blocks[:, None] * block_size +
                          torch.arange(block_size, device=q.device)[None, :]).reshape(-1)
            remote_abs = remote_abs[(remote_abs >= key_offset) &
                                    (remote_abs < key_end) &
                                    (remote_abs < local_start)]
            remote_idx = remote_abs - key_offset
        else:
            remote_idx = torch.empty(0, device=q.device, dtype=torch.long)

        local_abs_start = max(local_start, key_offset)
        local_abs_end = min(abs_q_end, key_end)
        local_idx = (torch.arange(local_abs_start, max(local_abs_start, local_abs_end),
                                  device=q.device) - key_offset)
        key_idx = torch.cat((remote_idx, local_idx))
        if key_idx.numel() == 0:
            raise RuntimeError("block-sparse attention found no visible keys for a query block")

        q_positions = torch.arange(abs_q_start, abs_q_end, device=q.device)
        k_positions = key_idx + key_offset
        allowed = ((k_positions[None, :] <= q_positions[:, None]) &
                   ((k_positions[None, :] >= q_positions[:, None] - local_window + 1) |
                    (k_positions[None, :] < local_start)))
        chunks.append(F.scaled_dot_product_attention(
            qh[:, :, q_start:q_end],
            kh.index_select(2, key_idx),
            vh.index_select(2, key_idx),
            attn_mask=allowed[None, None, :, :],
            dropout_p=0.0,
            is_causal=False,
        ))
        q_start = q_end

    return torch.cat(chunks, dim=2).transpose(1, 2).contiguous()


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, hd = cfg.d_model, cfg.head_dim
        self.n_heads, self.n_kv, self.hd = cfg.n_heads, cfg.n_kv_heads, hd
        self.backend = cfg.attention
        self.sparse_block_size = cfg.sparse_block_size
        self.local_window = cfg.local_window
        self.sparse_top_blocks = cfg.sparse_top_blocks
        self.sparse_block_mask = (
            _make_sparse_block_mask(cfg.seq_len, cfg.sparse_block_size, cfg.local_window, cfg.sparse_top_blocks)
            if cfg.attention == "block_sparse" else None
        )
        self.q_proj = nn.Linear(d, self.n_heads * hd, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv * hd, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv * hd, bias=False)
        self.out_proj = nn.Linear(self.n_heads * hd, d, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_value=None,
        *,
        use_cache: bool = False,
        position_offset: int = 0,
        key_position_offset: int = 0,
    ):
        B, S, _ = x.shape
        qkv = F.linear(x, torch.cat((self.q_proj.weight, self.k_proj.weight, self.v_proj.weight), dim=0))
        q_raw, k_raw, v_raw = qkv.split(
            (self.n_heads * self.hd, self.n_kv * self.hd, self.n_kv * self.hd), dim=-1,
        )
        q = q_raw.view(B, S, self.n_heads, self.hd).transpose(1, 2)
        k = k_raw.view(B, S, self.n_kv, self.hd).transpose(1, 2)
        v = v_raw.view(B, S, self.n_kv, self.hd).transpose(1, 2)
        q = apply_rope(q, cos[:S], sin[:S])
        k = apply_rope(k, cos[:S], sin[:S])
        if past_key_value is not None:
            past_k, past_v = past_key_value
            if past_k.shape[:2] != (B, self.n_kv) or past_v.shape[:2] != (B, self.n_kv):
                raise ValueError("past KV batch/head dimensions do not match this attention layer")
            if past_k.shape[-1] != self.hd or past_v.shape != past_k.shape:
                raise ValueError("past KV head dimensions do not match this attention layer")
            if past_k.device != k.device or past_k.dtype != k.dtype:
                raise ValueError("past KV device/dtype must match current attention activations")
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        present = (k, v) if use_cache else None
        use_turing = getattr(self, "backend", "efficient") == "flash_turing"
        if self.backend == "block_sparse":
            max_position = max(position_offset + S, key_position_offset + k.shape[2])
            needed_blocks = (max_position + self.sparse_block_size - 1) // self.sparse_block_size
            if needed_blocks > self.sparse_block_mask.shape[0]:
                self.sparse_block_mask = _make_sparse_block_mask(
                    max_position, self.sparse_block_size, self.local_window, self.sparse_top_blocks
                )
            y = _block_sparse_attention(
                q.transpose(1, 2).contiguous(),
                k.transpose(1, 2).contiguous(),
                v.transpose(1, 2).contiguous(),
                self.sparse_block_mask,
                self.sparse_block_size,
                self.local_window,
                position_offset,
                key_position_offset,
            ).transpose(1, 2)
        # The upstream SM75 kernel only exposes a causal flag, not absolute
        # query/key offsets. Use it only for aligned full-sequence attention;
        # cached decode and bounded replay must use the offset-aware SDPA path.
        elif (
            use_turing and x.is_cuda and q.dtype == torch.float16
            and self.hd in (64, 96, 128) and position_offset == 0
            and key_position_offset == 0 and S == k.shape[2]
        ):
            if torch.cuda.get_device_capability(x.device) != (7, 5):
                raise RuntimeError("flash_turing is built for Turing SM75 GPUs such as the NVIDIA T4.")
            # ssiu's SM75 kernel accepts native GQA and uses BSHD layout.
            y = _FlashAttentionTuring.apply(
                q.transpose(1, 2).contiguous(),
                k.transpose(1, 2).contiguous(),
                v.transpose(1, 2).contiguous(),
            ).transpose(1, 2)
        elif use_turing and x.is_cuda and (q.dtype != torch.float16 or self.hd not in (64, 96, 128)):
            if q.dtype != torch.float16 or self.hd not in (64, 96, 128):
                raise RuntimeError(
                    "flash_turing requires FP16 activations and head_dim 64, 96, or 128; "
                    f"got attention dtype={q.dtype}, head_dim={self.hd}."
                )
            raise RuntimeError("flash_turing is built for Turing SM75 GPUs such as the NVIDIA T4.")
        else:
            enable_gqa = self.n_kv != self.n_heads
            if position_offset == 0 and key_position_offset == 0 and S == k.shape[2]:
                try:
                    y = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True,
                        enable_gqa=enable_gqa,
                    )
                except (TypeError, RuntimeError):
                    # Some SDPA kernels do not implement native GQA; keep a
                    # compatible fallback while retaining the native path where supported.
                    if not enable_gqa:
                        raise
                    rep = self.n_heads // self.n_kv
                    y = F.scaled_dot_product_attention(
                        q, k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1),
                        attn_mask=None, dropout_p=0.0, is_causal=True,
                    )
            else:
                q_positions = position_offset + torch.arange(S, device=x.device)
                k_positions = key_position_offset + torch.arange(k.shape[2], device=x.device)
                causal_mask = k_positions[None, :] <= q_positions[:, None]
                try:
                    y = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=causal_mask[None, None], dropout_p=0.0,
                        is_causal=False, enable_gqa=enable_gqa,
                    )
                except (TypeError, RuntimeError):
                    if not enable_gqa:
                        raise
                    rep = self.n_heads // self.n_kv
                    y = F.scaled_dot_product_attention(
                        q, k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1),
                        attn_mask=causal_mask[None, None], dropout_p=0.0, is_causal=False,
                    )
        y = y.transpose(1, 2).contiguous().view(B, S, -1)
        y = self.out_proj(y)
        return (y, present) if use_cache else y


class SparseAttention(nn.Module):
    """Experimental causal top-k attention with chunked index and value gathers."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, hd = cfg.d_model, cfg.head_dim
        self.n_heads, self.n_kv, self.hd = cfg.n_heads, cfg.n_kv_heads, hd
        self.k = cfg.sparse_top_k
        self.q_proj = nn.Linear(d, self.n_heads * hd, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv * hd, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv * hd, bias=False)
        self.out_proj = nn.Linear(self.n_heads * hd, d, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        qkv = F.linear(x, torch.cat((self.q_proj.weight, self.k_proj.weight, self.v_proj.weight), dim=0))
        q_raw, k_raw, v_raw = qkv.split(
            (self.n_heads * self.hd, self.n_kv * self.hd, self.n_kv * self.hd), dim=-1,
        )
        q = q_raw.view(B, S, self.n_heads, self.hd).transpose(1, 2)
        k = k_raw.view(B, S, self.n_kv, self.hd).transpose(1, 2)
        v = v_raw.view(B, S, self.n_kv, self.hd).transpose(1, 2)
        q = apply_rope(q, cos[:S], sin[:S])
        k = apply_rope(k, cos[:S], sin[:S])
        if self.n_kv != self.n_heads:
            rep = self.n_heads // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        # Select past keys in bounded query chunks. Selection is an indexer only;
        # attention weights are recomputed differentiably below.
        top_k = min(self.k, S)
        chunk = min(32, S)
        out_chunks = []
        for start in range(0, S, chunk):
            end = min(start + chunk, S)
            q_chunk = q[:, :, start:end, :]
            positions = torch.arange(start, end, device=x.device).view(1, 1, -1, 1)
            with torch.no_grad():
                scores = torch.matmul(q_chunk.float(), k.float().transpose(-2, -1)) / math.sqrt(self.hd)
                key_positions = torch.arange(S, device=x.device).view(1, 1, 1, -1)
                scores.masked_fill_(key_positions > positions, float("-inf"))
                top_idx = torch.topk(scores, top_k, dim=-1, sorted=False).indices

            def attend(qc, kc, vc, indices, qpos):
                index_exp = indices.unsqueeze(-1).expand(-1, -1, -1, -1, self.hd)
                ks = torch.gather(kc.unsqueeze(3).expand(-1, -1, -1, top_k, -1), 2, index_exp)
                vs = torch.gather(vc.unsqueeze(3).expand(-1, -1, -1, top_k, -1), 2, index_exp)
                logits = (qc.unsqueeze(-2) * ks).sum(dim=-1) / math.sqrt(self.hd)
                valid = indices <= qpos
                logits = logits.masked_fill(~valid, float("-inf"))
                weights = F.softmax(logits.float(), dim=-1).to(vs.dtype)
                return (weights.unsqueeze(-1) * vs).sum(dim=-2)

            # Recompute the large gathered tensors during backward instead of
            # retaining O(batch * heads * sequence * top_k * head_dim) activations.
            if self.training and torch.is_grad_enabled():
                out_chunk = checkpoint(attend, q_chunk, k, v, top_idx, positions, use_reentrant=False)
            else:
                out_chunk = attend(q_chunk, k, v, top_idx, positions)
            out_chunks.append(out_chunk)
        out_sparse = torch.cat(out_chunks, dim=2)
        out_sparse = out_sparse.transpose(1, 2).contiguous().view(B, S, -1)
        return self.out_proj(out_sparse)


# --------------------------------------------------------------------------- #
# Hashed n-gram lookup memory (prime-modular hash, configurable table sizes)
# --------------------------------------------------------------------------- #
def find_next_prime(start: int, seen_primes: set) -> int:
    """Find next prime > start not in seen_primes."""
    candidate = start + 1
    # Simple primality test; vocab sizes are small (<1M) so this is fine
    def is_prime(n: int) -> bool:
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        r = int(n ** 0.5)
        for i in range(3, r + 1, 2):
            if n % i == 0:
                return False
        return True
    while True:
        if is_prime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


class NgramHashMapping:
    """Prime-modular N-gram hash mapping per layer (O(1) lookup)."""
    def __init__(
        self,
        vocab_size_per_ngram: List[int],
        max_ngram_size: int,
        layer_id: int,
        seed: int = 0,
        pad_id: int = 0,
    ):
        self.vocab_size_per_ngram = vocab_size_per_ngram
        self.max_ngram_size = max_ngram_size
        self.layer_id = layer_id
        self.pad_id = pad_id

        # Per-layer multipliers (random odd ints derived from seed + layer)
        base_seed = seed + 10007 * layer_id
        rng = np.random.default_rng(base_seed)
        # Small random multipliers; half-bound keeps hashes well-distributed
        half_bound = max(1, 1000000)
        r = rng.integers(low=0, high=half_bound, size=(max_ngram_size,))
        self.multipliers = r * 2 + 1  # ensure odd

        # Compute vocab sizes per head per ngram
        self.vocab_size_across_layers = {}
        self.primes_per_layer = {}
        vocab_sizes = self._compute_vocab_sizes()

    def _compute_vocab_sizes(self):
        seen_primes = set()
        vocab_sizes = {}
        primes_map = {}
        # We build a flat list for MultiHeadEmbedding: each head of each ngram
        sizes = []
        layer_primes = {}
        for n in range(2, self.max_ngram_size + 1):
            idx = n - 2
            vocab_size = self.vocab_size_per_ngram[idx] if idx < len(self.vocab_size_per_ngram) else 128000
            # Four independent hash heads for each n-gram order.
            num_heads = 4
            head_sizes = []
            head_primes = []
            for _ in range(num_heads):
                prime = find_next_prime(vocab_size - 1, seen_primes)
                seen_primes.add(prime)
                head_sizes.append(prime)
                head_primes.append(prime)
            sizes.extend(head_sizes)
            layer_primes[n] = head_primes
        vocab_sizes[self.layer_id] = sizes
        self.vocab_size_across_layers[self.layer_id] = sizes
        self.primes_per_layer[self.layer_id] = layer_primes
        return vocab_sizes

    def hash(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return hash ids [B, S, H] where H = sum(head_sizes) for all ngrams."""
        x = input_ids.long()
        B, T = x.shape
        all_hashes = []
        layer_primes = self.primes_per_layer.get(self.layer_id, {})
        for n in range(2, self.max_ngram_size + 1):
            # Build shifted windows
            shifted = [x]
            for k in range(1, n):
                pad = torch.full((B, k), self.pad_id, dtype=torch.long, device=x.device)
                shifted_k = torch.cat([pad, x[:, :-k] if k > 0 else x], dim=1)
                shifted.append(shifted_k)
            # Combine with multipliers using integer XOR (pure torch)
            mix = shifted[0].long() * int(self.multipliers[0])
            for k in range(1, n):
                mix_k = shifted[k].long() * int(self.multipliers[k])
                mix = torch.bitwise_xor(mix, mix_k)
            # Per-head modular hash using the same primes as init
            primes_for_n = layer_primes.get(n, [])
            for h in range(len(primes_for_n)):
                prime = primes_for_n[h]
                head_hash = mix % prime
                all_hashes.append(head_hash)
        # Stack: [B, S, H_total]
        if all_hashes:
            return torch.stack(all_hashes, dim=-1)
        return torch.zeros(B, T, 0, dtype=torch.long, device=x.device)


class MultiHeadEmbedding(nn.Module):
    """Embedding table split across hashed n-gram heads."""
    def __init__(self, vocab_sizes: List[int], embed_dim: int):
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.embed_dim = embed_dim
        offsets = [0]
        for v in vocab_sizes[:-1]:
            offsets.append(offsets[-1] + v)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        total_n = sum(vocab_sizes)
        # Each hash head owns one non-overlapping vocabulary range.
        self.embedding = nn.Embedding(num_embeddings=total_n, embedding_dim=embed_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, S, H]
        shifted = input_ids + self.offsets
        return self.embedding(shifted).flatten(start_dim=-2, end_dim=-1)


class Engram(nn.Module):
    """Hashed n-gram lookup followed by gated residual fusion."""
    def __init__(self, cfg: ModelConfig, layer_id: int = 0):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.pad_id = cfg.pad_token_id

        # Configurable vocab sizes per ngram order; avoids vocab_size**2 demon
        # Default: [4*vocab, 4*vocab] for 2-gram / 3-gram at reasonable scale
        vocab_per_ngram = cfg.engram_vocab_size if hasattr(cfg, 'engram_vocab_size') else [
            max(128000, cfg.vocab_size * 4),
            max(128000, cfg.vocab_size * 4),
        ]
        self.hash_mapping = NgramHashMapping(
            vocab_size_per_ngram=vocab_per_ngram,
            max_ngram_size=cfg.engram_ngram,
            layer_id=layer_id,
            seed=cfg.seed if hasattr(cfg, 'seed') else 1337,
            pad_id=self.pad_id,
        )

        # Embedding per head; total dims = n_embed_per_ngram
        embed_per_head = cfg.engram_table_dim // 4  # 4 heads by default in mapping
        vocab_sizes_for_layer = self.hash_mapping.vocab_size_across_layers.get(layer_id, [])
        if not vocab_sizes_for_layer:
            vocab_sizes_for_layer = [max(32000, cfg.vocab_size)] * 8
        self.embedding = MultiHeadEmbedding(
            vocab_sizes=vocab_sizes_for_layer,
            embed_dim=max(16, embed_per_head),
        )
        total_embed_dim = len(vocab_sizes_for_layer) * max(16, embed_per_head)

        # Fusion projections
        self.fuse_proj = nn.Linear(total_embed_dim, cfg.d_model, bias=False)
        self.gate_proj = nn.Linear(cfg.d_model * 2, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        idx: torch.Tensor,
        prefix_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: [B, S, D] hidden states; idx: [B, S] token ids
        if prefix_ids is not None and prefix_ids.shape[1]:
            hash_input = torch.cat((prefix_ids, idx), dim=1)
            hash_ids = self.hash_mapping.hash(hash_input)[:, -idx.shape[1]:]
        else:
            hash_ids = self.hash_mapping.hash(idx)  # [B, S, H]
        emb = self.embedding(hash_ids)            # [B, S, H*D]
        # Project to model dim
        proj = self.fuse_proj(emb)              # [B, S, D]
        # Gate fusion: concat(hidden, proj) -> sigmoid gate
        gate_input = torch.cat([x, proj], dim=-1)
        g = torch.sigmoid(self.gate_proj(gate_input))
        return x + g * proj


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep the original parameters/checkpoint keys, but issue one wider GEMM.
        gate_up = F.linear(x, torch.cat((self.gate_proj.weight, self.up_proj.weight), dim=0))
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


@dataclass
class GDNState:
    """Streaming state for one Gated DeltaNet layer."""

    conv: torch.Tensor       # [B, kernel_size - 1, 3 * key/value width]
    recurrent: torch.Tensor  # [B, heads, key_dim, value_dim], fp32 state


class GatedDeltaNet(nn.Module):
    """Qwen-style Gated DeltaNet with FLA chunk kernel and a torch reference.

    The recurrent update is the delta rule from Qwen3.5: decay the matrix state,
    predict the value from the key, then write the prediction residual. The
    torch path is for correctness and CPU use; install flash-linear-attention
    for practical GPU pretraining performance.
    """

    _warned_missing_fla = False

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.hidden_size = cfg.d_model
        self.num_heads = cfg.gdn_num_heads or cfg.n_heads
        self.head_dim = cfg.gdn_head_dim or cfg.head_dim
        self.width = self.num_heads * self.head_dim
        self.conv_kernel_size = cfg.gdn_conv_kernel_size
        self.backend = cfg.gdn_backend

        self.in_proj_qkv = nn.Linear(self.hidden_size, 3 * self.width, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.width, bias=False)
        self.in_proj_beta = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.conv1d = nn.Conv1d(
            3 * self.width, 3 * self.width, self.conv_kernel_size,
            groups=3 * self.width, bias=False,
        )
        self.A_log = nn.Parameter(torch.empty(self.num_heads))
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))
        self.norm_weight = nn.Parameter(torch.ones(self.head_dim))
        self.norm_eps = cfg.norm_eps
        self.out_proj = nn.Linear(self.width, self.hidden_size, bias=False)

    def _torch_recurrence(self, q, k, v, decay, beta, initial_state):
        # The reference follows Qwen's recurrent implementation in fp32.
        output_dtype = v.dtype
        q, k, v, decay, beta = [t.float() for t in (q, k, v, decay, beta)]
        q = F.normalize(q, p=2, dim=-1, eps=1e-6) / math.sqrt(self.head_dim)
        k = F.normalize(k, p=2, dim=-1, eps=1e-6)
        state = (
            initial_state.float() if initial_state is not None else
            torch.zeros(q.shape[0], self.num_heads, self.head_dim, self.head_dim,
                        device=q.device, dtype=torch.float32)
        )
        outputs = []
        for i in range(q.shape[1]):
            kt = k[:, i]
            state = state * decay[:, i].exp()[:, :, None, None]
            predicted = (state * kt.unsqueeze(-1)).sum(dim=-2)
            delta = (v[:, i] - predicted) * beta[:, i, :, None]
            state = state + kt.unsqueeze(-1) * delta.unsqueeze(-2)
            outputs.append((state * q[:, i].unsqueeze(-1)).sum(dim=-2))
        return torch.stack(outputs, dim=1).to(output_dtype), state

    def forward(self, x: torch.Tensor, past_state: Optional[GDNState] = None,
                use_cache: bool = False):
        batch, seq_len, _ = x.shape
        mixed = self.in_proj_qkv(x)
        if past_state is not None:
            if past_state.conv.shape[0] != batch or past_state.conv.shape[2] != mixed.shape[-1]:
                raise ValueError("GDN convolution cache shape does not match this layer")
            conv_input = torch.cat((past_state.conv, mixed), dim=1)
        else:
            conv_input = F.pad(mixed, (0, 0, self.conv_kernel_size - 1, 0))
        convolved = self.conv1d(conv_input.transpose(1, 2)).transpose(1, 2)
        convolved = F.silu(convolved[:, -seq_len:])
        q, k, v = convolved.split(self.width, dim=-1)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim)
        beta = self.in_proj_beta(x).sigmoid()
        a = self.in_proj_a(x)
        decay = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())

        use_fla = self.backend != "torch" and x.is_cuda
        fla_op = None
        if use_fla:
            try:
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule
                fla_op = chunk_gated_delta_rule
            except Exception as exc:  # noqa: BLE001
                if self.backend == "fla":
                    raise RuntimeError(
                        "gdn_backend='fla' could not import flash-linear-attention; "
                        f"check its PyTorch/CUDA binary dependencies ({exc})"
                    ) from exc
                if not GatedDeltaNet._warned_missing_fla:
                    warnings.warn(
                        "flash-linear-attention failed to import; GDN is using the slow PyTorch "
                        f"reference recurrence ({exc}). Check the FLA and PyTorch binary versions.",
                        RuntimeWarning,
                    )
                    GatedDeltaNet._warned_missing_fla = True
        elif self.backend == "fla":
            raise RuntimeError("gdn_backend='fla' requires CUDA and flash-linear-attention")

        initial = None if past_state is None else past_state.recurrent
        if fla_op is not None:
            core, recurrent = fla_op(
                q, k, v, g=decay, beta=beta,
                initial_state=initial,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
            )
            if use_cache and recurrent is None:
                raise RuntimeError("Flash Linear Attention did not return the requested GDN state")
        else:
            core, recurrent = self._torch_recurrence(q, k, v, decay, beta, initial)

        z = self.in_proj_z(x).view(batch, seq_len, self.num_heads, self.head_dim)
        core = core.float()
        core = core * torch.rsqrt(core.square().mean(dim=-1, keepdim=True) + self.norm_eps)
        core = core * self.norm_weight.float() * F.silu(z.float())
        output = self.out_proj(core.to(x.dtype).reshape(batch, seq_len, self.width))
        if not use_cache:
            return output
        conv_cache = conv_input[:, -(self.conv_kernel_size - 1):] if self.conv_kernel_size > 1 else conv_input[:, :0]
        if recurrent is None:
            raise RuntimeError("GDN backend failed to return recurrent state for cache")
        return output, GDNState(conv_cache, recurrent)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_id: int = 0):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.is_gdn = bool(cfg.hybrid_gdn and layer_id % 4 != 0)
        if self.is_gdn:
            self.gdn = GatedDeltaNet(cfg)
            self.attn = None
        elif cfg.attention == "sparse":
            self.attn = SparseAttention(cfg)
            self.gdn = None
        else:
            self.attn = Attention(cfg)
            self.gdn = None
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)
        # Memory lookup is limited to the first four layers.
        self.engram = (Engram(cfg, layer_id=layer_id) if cfg.use_engram and layer_id < 4 else None)
        self.layer_id = layer_id

    def forward(
        self,
        x,
        cos,
        sin,
        idx=None,
        *,
        past_key_value=None,
        use_cache: bool = False,
        position_offset: int = 0,
        key_position_offset: int = 0,
        engram_prefix: Optional[torch.Tensor] = None,
    ):
        normalized = self.attn_norm(x)
        if self.gdn is not None:
            if use_cache:
                gdn_out, present = self.gdn(
                    normalized, past_state=past_key_value, use_cache=True,
                )
            else:
                gdn_out = self.gdn(normalized)
            attn_out = gdn_out
        elif isinstance(self.attn, SparseAttention):
            if use_cache:
                raise ValueError("KV caching is not implemented for the legacy experimental top-k backend")
            attn_out = self.attn(normalized, cos, sin)
        else:
            attn_result = self.attn(
                normalized, cos, sin, past_key_value,
                use_cache=use_cache,
                position_offset=position_offset,
                key_position_offset=key_position_offset,
            )
            if use_cache:
                attn_out, present = attn_result
            else:
                attn_out = attn_result
        x = x + attn_out
        if self.engram is not None and idx is not None:
            x = self.engram(x, idx, engram_prefix)
        x = x + self.mlp(self.mlp_norm(x))
        return (x, present) if use_cache else x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([
            Block(cfg, layer_id=i) for i in range(cfg.n_layers)
        ])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self.gradient_checkpointing = False
        self._init_weights()
        cos, sin = rope_cos_sin(cfg.seq_len, cfg.head_dim, cfg.rope_theta, torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
        rs = self.cfg.init_std / math.sqrt(2.0 * self.cfg.n_layers)
        for block in self.blocks:
            if block.gdn is not None:
                nn.init.normal_(block.gdn.conv1d.weight, std=self.cfg.init_std)
                nn.init.ones_(block.gdn.norm_weight)
                nn.init.ones_(block.gdn.dt_bias)
                with torch.no_grad():
                    block.gdn.A_log.copy_(torch.empty_like(block.gdn.A_log).uniform_(0.01, 16).log_())
            if hasattr(block.attn, 'out_proj'):
                nn.init.normal_(block.attn.out_proj.weight, std=rs)
            nn.init.normal_(block.mlp.down_proj.weight, std=rs)
        nn.init.ones_(self.norm_f.weight)

    def _ensure_rope(self, seq_len: int, device: torch.device) -> None:
        if seq_len > self.rope_cos.shape[0] or self.rope_cos.device != device:
            cos, sin = rope_cos_sin(seq_len, self.cfg.head_dim, self.cfg.rope_theta, device)
            self.rope_cos, self.rope_sin = cos, sin

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        last_only: bool = False,
        position_offset: Optional[int] = None,
        *,
        past_key_values: Optional[KVCache] = None,
        use_cache: bool = False,
        max_cache_length: Optional[int] = None,
        cache_token_ids: Optional[torch.Tensor] = None,
        engram_prefix: Optional[torch.Tensor] = None,
        kv_chunk_size: int = 0,
    ):
        if kv_chunk_size:
            if kv_chunk_size < 1:
                raise ValueError("kv_chunk_size must be positive")
            if targets is None:
                raise ValueError("kv_chunk_size is a training forward option and requires targets")
            if self.gradient_checkpointing:
                raise ValueError("kv_chunk_size cannot be combined with gradient checkpointing")
            if self.cfg.attention == "sparse":
                raise ValueError("kv_chunk_size is unsupported for the legacy experimental top-k backend")
            if idx.shape[1] == 0:
                raise ValueError("kv_chunk_size requires a non-empty input sequence")
            if cache_token_ids is not None:
                raise ValueError("cache_token_ids is unsupported by the internal kv_chunk_size training path")
            if past_key_values is None and position_offset not in (None, 0):
                raise ValueError("kv_chunk_size without a past cache must start at position zero")
            hidden_chunks = []
            loss_total = None
            cache = past_key_values
            for chunk_start in range(0, idx.shape[1], kv_chunk_size):
                chunk_end = min(chunk_start + kv_chunk_size, idx.shape[1])
                chunk_offset = None if cache is not None else (
                    (position_offset or 0) + chunk_start
                )
                chunk_prefix = engram_prefix if cache is None else None
                hidden, chunk_loss, cache = self.forward(
                    idx[:, chunk_start:chunk_end],
                    targets=targets[:, chunk_start:chunk_end],
                    position_offset=chunk_offset,
                    past_key_values=cache,
                    use_cache=True,
                    max_cache_length=max_cache_length,
                    cache_token_ids=cache_token_ids if cache is None else None,
                    engram_prefix=chunk_prefix,
                )
                hidden_chunks.append(hidden)
                scaled_loss = chunk_loss.float() * (chunk_end - chunk_start)
                loss_total = scaled_loss if loss_total is None else loss_total + scaled_loss
            hidden = torch.cat(hidden_chunks, dim=1)
            loss = loss_total / idx.shape[1]
            return (hidden, loss, cache) if use_cache else (hidden, loss)
        if use_cache and self.training and self.gradient_checkpointing:
            raise ValueError("KV-cache forward cannot be combined with gradient checkpointing")
        if use_cache and self.cfg.attention == "sparse":
            raise ValueError("KV caching is not implemented for the legacy experimental top-k backend")
        if max_cache_length is not None and max_cache_length < 1:
            raise ValueError("max_cache_length must be positive")
        if past_key_values is not None and not use_cache:
            raise ValueError("past_key_values requires use_cache=True")
        if past_key_values is not None:
            if not isinstance(past_key_values, KVCache):
                raise TypeError("past_key_values must be a KVCache returned by this model")
            if len(past_key_values.layers) != self.cfg.n_layers:
                raise ValueError("KV cache layer count does not match model")
            if past_key_values.token_ids.shape[0] != idx.shape[0]:
                raise ValueError("KV cache batch size does not match input")
            if past_key_values.token_ids.device != idx.device:
                raise ValueError("KV cache token history must be on the same device as input")
            if not 0 <= past_key_values.start_position <= past_key_values.position:
                raise ValueError("KV cache has an invalid starting position")
            expected_cached_length = past_key_values.position - past_key_values.start_position
            for block, layer_state in zip(self.blocks, past_key_values.layers):
                if block.gdn is not None:
                    if not isinstance(layer_state, GDNState):
                        raise TypeError("GDN layer cache has the wrong state type")
                    if layer_state.conv.shape[0] != idx.shape[0]:
                        raise ValueError("GDN convolution cache batch does not match input")
                    expected_recurrent = (
                        idx.shape[0], block.gdn.num_heads, block.gdn.head_dim, block.gdn.head_dim,
                    )
                    if tuple(layer_state.recurrent.shape) != expected_recurrent:
                        raise ValueError("GDN recurrent cache shape does not match layer configuration")
                    if layer_state.conv.device != idx.device or layer_state.recurrent.device != idx.device:
                        raise ValueError("GDN cache tensors must be on the same device as input")
                else:
                    if not isinstance(layer_state, tuple) or len(layer_state) != 2:
                        raise TypeError("attention layer cache must contain a key/value pair")
                    layer_k, layer_v = layer_state
                    if layer_k.shape[0] != idx.shape[0] or layer_v.shape != layer_k.shape:
                        raise ValueError("KV cache tensor shape does not match batch")
                    if layer_k.shape[2] != expected_cached_length:
                        raise ValueError("KV cache tensors do not align with their absolute positions")
            expected_offset = past_key_values.position
            if position_offset is not None and position_offset != expected_offset:
                raise ValueError("position_offset must equal the cached token count")
            position_offset = expected_offset
            key_position_offset = past_key_values.start_position
        else:
            position_offset = 0 if position_offset is None else position_offset
            key_position_offset = position_offset
        if position_offset < 0:
            raise ValueError("position_offset must be non-negative")
        if past_key_values is not None and cache_token_ids is not None:
            raise ValueError("cache_token_ids is only used to initialize a cache without past_key_values")
        if use_cache and past_key_values is None and position_offset > 0 and cache_token_ids is None:
            raise ValueError("cache initialization at a non-zero position requires the full cache_token_ids history")
        self._ensure_rope(position_offset + idx.shape[1], idx.device)
        cos = self.rope_cos[position_offset:position_offset + idx.shape[1]]
        sin = self.rope_sin[position_offset:position_offset + idx.shape[1]]
        if engram_prefix is None and past_key_values is not None and self.cfg.use_engram:
            lookback = max(0, self.cfg.engram_ngram - 1)
            engram_prefix = past_key_values.token_ids[:, -lookback:] if lookback else None
        elif engram_prefix is not None:
            lookback = max(0, self.cfg.engram_ngram - 1)
            engram_prefix = engram_prefix[:, -lookback:] if lookback else None

        if use_cache:
            present_layers = []
        x = self.tok_emb(idx)
        for layer_id, block in enumerate(self.blocks):
            prefix = engram_prefix
            if block.engram is not None:
                if prefix is not None and prefix.shape[0] != idx.shape[0]:
                    raise ValueError("Engram prefix batch size does not match input")
            if use_cache:
                past_layer = None if past_key_values is None else past_key_values.layers[layer_id]
                x, present = block(
                    x, cos, sin, idx,
                    past_key_value=past_layer,
                    use_cache=True,
                    position_offset=position_offset,
                    key_position_offset=key_position_offset,
                    engram_prefix=prefix,
                )
                if (max_cache_length is not None and block.gdn is None
                        and present[0].shape[2] > max_cache_length):
                    present = (present[0][:, :, -max_cache_length:], present[1][:, :, -max_cache_length:])
                present_layers.append(present)
            elif self.gradient_checkpointing and self.training and x.requires_grad:
                args = (x, cos, sin, idx if block.engram is not None else None)
                x = checkpoint(block, *args, use_reentrant=False)
            else:
                x = block(
                    x, cos, sin, idx if block.engram is not None else None,
                    engram_prefix=prefix,
                )
        x = self.norm_f(x)

        new_cache = None
        if use_cache:
            if past_key_values is None:
                token_history = idx if cache_token_ids is None else cache_token_ids.to(idx.device)
                if token_history.ndim != 2 or token_history.shape[0] != idx.shape[0]:
                    raise ValueError("cache_token_ids must have shape [batch, tokens]")
                if token_history.shape[1] != position_offset + idx.shape[1]:
                    raise ValueError("cache_token_ids must end at the supplied absolute position")
            else:
                token_history = torch.cat((past_key_values.token_ids, idx), dim=1)
            first_attention_state = next(
                state for block, state in zip(self.blocks, present_layers) if block.gdn is None
            )
            key_length = first_attention_state[0].shape[2]
            new_start = position_offset + idx.shape[1] - key_length
            new_cache = KVCache(tuple(present_layers), token_history, new_start)

        if targets is not None:
            result = (x, self._loss(x, targets))
            return (*result, new_cache) if use_cache else result
        if last_only:
            result = (self.lm_head(x[:, -1:, :]), None)
        else:
            result = (self.lm_head(x), None)
        return (*result, new_cache) if use_cache else result

    def _loss(self, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total = hidden.new_zeros((), dtype=torch.float32)
        for h, t in zip(hidden.split(self.cfg.loss_chunk, dim=1), targets.split(self.cfg.loss_chunk, dim=1)):
            logits = self.lm_head(h)
            total = total + F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), t.reshape(-1).long(), reduction="sum"
            )
        return total / targets.numel()

    def set_gradient_checkpointing(self, on: bool) -> None:
        self.gradient_checkpointing = bool(on)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 0.8,
        top_k: int = 50,
        eos_id: Optional[int] = None,
        device=None,
        swa_bounded_replay: bool = False,
    ) -> torch.Tensor:
        prev_training = self.training
        self.eval()
        try:
            device = device or idx.device
            idx = idx.to(device)
            if swa_bounded_replay:
                if self.cfg.attention != "block_sparse":
                    raise ValueError("SWA bounded replay requires a block_sparse checkpoint/config")
                replay_window = self.cfg.local_window
            if max_new_tokens <= 0:
                return idx

            use_kv_cache = self.cfg.attention != "sparse"
            if swa_bounded_replay:
                replay_start = max(0, idx.shape[1] - replay_window)
                if self.cfg.hybrid_gdn:
                    # GDN carries a recurrent state over the whole prefix; a
                    # suffix-only replay would silently reset that memory.
                    # Prefill the full prompt, but bound attention KV storage.
                    logits, _, cache = self.forward(
                        idx, last_only=True, use_cache=True,
                        max_cache_length=replay_window,
                    )
                else:
                    replay_tokens = idx[:, replay_start:]
                    lookback = max(0, self.cfg.engram_ngram - 1) if self.cfg.use_engram else 0
                    engram_prefix = idx[:, max(0, replay_start - lookback):replay_start] if lookback else None
                    logits, _, cache = self.forward(
                        replay_tokens,
                        last_only=True,
                        position_offset=replay_start,
                        use_cache=True,
                        max_cache_length=replay_window,
                        cache_token_ids=idx,
                        engram_prefix=engram_prefix,
                    )
            elif use_kv_cache:
                logits, _, cache = self.forward(idx, last_only=True, use_cache=True)
            else:
                logits, _ = self.forward(idx, last_only=True)

            next_tok = idx.clone()
            for step in range(max_new_tokens):
                logits = logits[:, -1, :].float()
                if temperature <= 0:
                    y = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / max(1e-6, temperature)
                    if top_k and top_k < logits.size(-1):
                        v, _ = torch.topk(logits, top_k)
                        logits[logits < v[..., [-1]]] = -float("inf")
                    probs = torch.softmax(logits, dim=-1)
                    y = torch.multinomial(probs, num_samples=1)
                if eos_id is not None and bool((y == eos_id).any(dim=1, keepdim=True).all()):
                    break
                next_tok = torch.cat([next_tok, y], dim=1)
                if step + 1 < max_new_tokens and use_kv_cache:
                    logits, _, cache = self.forward(
                        y,
                        last_only=True,
                        past_key_values=cache,
                        use_cache=True,
                        max_cache_length=replay_window if swa_bounded_replay else None,
                    )
                elif step + 1 < max_new_tokens:
                    logits, _ = self.forward(next_tok, last_only=True)
            return next_tok
        finally:
            if prev_training:
                self.train()
            else:
                self.eval()


def unwrap(module) -> nn.Module:
    while True:
        if hasattr(module, "_orig_mod"):
            module = module._orig_mod
            continue
        if hasattr(module, "module"):
            module = module.module
            continue
        return module


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
