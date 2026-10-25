"""Python interface for FlashAttention Turing extension."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import flash_attn_turing as flash_attn_gpu
except ImportError:
    # The compact block-sparse PyTorch path can be used independently; the
    # fused dense causal path requires the separately compiled extension.
    flash_attn_gpu = None


def maybe_contiguous(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if flash_attn_gpu is None:
        raise RuntimeError("Build/install the flash_attn_turing CUDA extension first")
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, lse = flash_attn_gpu.fwd(
        q,
        k,
        v,
        softmax_scale,
        causal,
    )
    return out, lse


def _flash_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dout, q, k, v, out, lse = [maybe_contiguous(x) for x in (dout, q, k, v, out, lse)]
    dq, dk, dv = flash_attn_gpu.bwd(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        causal,
    )
    return dq, dk, dv


def _flash_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if flash_attn_gpu is None:
        raise RuntimeError("Build/install the flash_attn_turing CUDA extension first")
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, lse = flash_attn_gpu.varlen_fwd(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
    )
    return out, lse


def _flash_attn_varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dout, q, k, v, out, lse = [maybe_contiguous(x) for x in (dout, q, k, v, out, lse)]
    dq, dk, dv = flash_attn_gpu.varlen_bwd(
        q,
        k,
        v,
        out,
        lse,
        dout,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
    )
    return dq, dk, dv


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: Optional[float],
        causal: bool,
        is_grad_enabled: bool,
    ):
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_forward(q, k, v, softmax_scale, causal)
        is_grad = is_grad_enabled and any(x.requires_grad for x in (q, k, v))
        if is_grad:
            ctx.save_for_backward(q, k, v, out, lse)
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _flash_attn_backward(dout, q, k, v, out, lse, ctx.softmax_scale, ctx.causal)
        return dq, dk, dv, None, None, None


class FlashAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv: torch.Tensor,
        softmax_scale: Optional[float],
        causal: bool,
        is_grad_enabled: bool,
    ):
        q, k, v = (
            qkv.select(dim=2, index=0).contiguous(),
            qkv.select(dim=2, index=1).contiguous(),
            qkv.select(dim=2, index=2).contiguous(),
        )
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_forward(q, k, v, softmax_scale, causal)
        is_grad = is_grad_enabled and qkv.requires_grad
        if is_grad:
            ctx.save_for_backward(q, k, v, out, lse)
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _flash_attn_backward(dout, q, k, v, out, lse, ctx.softmax_scale, ctx.causal)
        qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
        dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
        dqkv[:, :, 0] = dq
        dqkv[:, :, 1] = dk
        dqkv[:, :, 2] = dv
        return dqkv, None, None, None


class FlashAttnKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        kv: torch.Tensor,
        softmax_scale: Optional[float],
        causal: bool,
        is_grad_enabled: bool,
    ):
        k, v = (
            kv.select(dim=2, index=0).contiguous(),
            kv.select(dim=2, index=1).contiguous(),
        )
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_forward(q, k, v, softmax_scale, causal)
        is_grad = is_grad_enabled and any(x.requires_grad for x in (q, kv))
        if is_grad:
            ctx.save_for_backward(q, k, v, out, lse)
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _flash_attn_backward(dout, q, k, v, out, lse, ctx.softmax_scale, ctx.causal)
        kv_shape = k.shape[:-2] + (2, *k.shape[-2:])
        dkv = torch.empty(kv_shape, dtype=k.dtype, device=k.device)
        dkv[:, :, 0] = dk
        dkv[:, :, 1] = dv
        return dq, dkv, None, None, None


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: Optional[float],
        causal: bool,
        is_grad_enabled: bool,
    ):
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            causal,
        )
        is_grad = is_grad_enabled and any(x.requires_grad for x in (q, k, v))
        if is_grad:
            ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        dq, dk, dv = _flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.softmax_scale,
            ctx.causal,
        )
        return dq, dk, dv, None, None, None, None, None, None, None


class FlashAttnVarlenQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        softmax_scale: Optional[float],
        causal: bool,
        is_grad_enabled: bool,
    ):
        q, k, v = (
            qkv.select(dim=1, index=0).contiguous(),
            qkv.select(dim=1, index=1).contiguous(),
            qkv.select(dim=1, index=2).contiguous(),
        )
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            softmax_scale,
            causal,
        )
        is_grad = is_grad_enabled and qkv.requires_grad
        if is_grad:
            ctx.save_for_backward(q, k, v, out, lse, cu_seqlens)
            ctx.max_seqlen = max_seqlen
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, lse, cu_seqlens = ctx.saved_tensors
        dq, dk, dv = _flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens,
            cu_seqlens,
            ctx.max_seqlen,
            ctx.max_seqlen,
            ctx.softmax_scale,
            ctx.causal,
        )
        qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
        dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
        dqkv[:, 0] = dq
        dqkv[:, 1] = dk
        dqkv[:, 2] = dv
        return dqkv, None, None, None, None, None


class FlashAttnVarlenKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        kv: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: Optional[float],
        causal: bool,
        is_grad_enabled: bool,
    ):
        k, v = (
            kv.select(dim=1, index=0).contiguous(),
            kv.select(dim=1, index=1).contiguous(),
        )
        softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        out, lse = _flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            causal,
        )
        is_grad = is_grad_enabled and any(x.requires_grad for x in (q, kv))
        if is_grad:
            ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        dq, dk, dv = _flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.softmax_scale,
            ctx.causal,
        )
        kv_shape = k.shape[:-2] + (2, *k.shape[-2:])
        dkv = torch.empty(kv_shape, dtype=k.dtype, device=k.device)
        dkv[:, 0] = dk
        dkv[:, 1] = dv
        return dq, dkv, None, None, None, None, None, None, None


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, headdim)
        k: (batch_size, seqlen_k, nheads_k, headdim)
        v: (batch_size, seqlen_k, nheads_k, headdim)
        causal: bool. Whether to apply causal attention mask.
    Return:
        out: (batch_size, seqlen_q, nheads, headdim).
    """
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        torch.is_grad_enabled(),
    )


@torch.compiler.disable
def flash_attn_block_sparse_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int,
    local_window: int,
    *,
    query_offset: int = 0,
    key_offset: int = 0,
) -> torch.Tensor:
    """Exact causal attention over selected KV blocks plus a local window.

    The current SM75 CUDA kernels do not support sparse tile lists. This
    reference path compacts each query block's selected KV blocks and lets
    PyTorch SDPA compute only over that compacted set. It preserves autograd
    and exact token-level causal/local masking, but has Python/gather overhead;
    it is a correctness-first reference path, not a fused sparse CUDA kernel.

    q: [batch, query_len, heads, dim]; k/v: [batch, key_len, kv_heads, dim]
    block_mask uses absolute query/key block indices; True includes that key block.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must be rank-4 BSHD tensors")
    if block_size < 1 or local_window < 1:
        raise ValueError("block_size and local_window must be positive")
    batch, seqlen, heads, dim = q.shape
    if k.shape[0] != batch or v.shape[:3] != k.shape[:3] or k.shape[-1] != dim or v.shape[-1] != dim:
        raise ValueError("q/k/v batch and head dimensions must match (sequence lengths may differ)")
    if query_offset < 0 or key_offset < 0:
        raise ValueError("query_offset and key_offset must be non-negative")
    if k.shape[2] != v.shape[2] or heads % k.shape[2]:
        raise ValueError("KV head counts must match and divide the query head count")
    expected_q_blocks = (query_offset + seqlen + block_size - 1) // block_size
    expected_k_blocks = (key_offset + k.shape[1] + block_size - 1) // block_size
    if block_mask.ndim != 2 or block_mask.shape[0] < expected_q_blocks or block_mask.shape[1] < expected_k_blocks:
        raise ValueError("block_mask must cover the absolute query and key positions")

    qh = q.transpose(1, 2)
    kh = k.transpose(1, 2)
    vh = v.transpose(1, 2)
    if heads != k.shape[2]:
        rep = heads // k.shape[2]
        kh = kh.repeat_interleave(rep, dim=1)
        vh = vh.repeat_interleave(rep, dim=1)

    outputs = []
    q_start = 0
    while q_start < seqlen:
        q_abs_start = query_offset + q_start
        q_block_id = q_abs_start // block_size
        q_count = min(block_size - (q_abs_start % block_size), seqlen - q_start)
        q_end = q_start + q_count
        q_abs_end = query_offset + q_end
        local_start = max(0, q_abs_start - local_window + 1)

        # Selected remote blocks must be wholly before the local interval.
        selected = block_mask[q_block_id].to(dtype=torch.bool).nonzero(as_tuple=False).flatten().to(q.device)
        if selected.numel():
            selected = selected[selected * block_size + block_size <= local_start]
            remote_abs = (selected[:, None] * block_size + torch.arange(block_size, device=q.device)[None, :]).reshape(-1)
            remote_abs = remote_abs[
                (remote_abs >= key_offset)
                & (remote_abs < key_offset + k.shape[1])
                & (remote_abs < local_start)
            ]
            remote_idx = remote_abs - key_offset
        else:
            remote_idx = torch.empty(0, device=q.device, dtype=torch.long)

        local_abs_start = max(local_start, key_offset)
        local_abs_end = min(q_abs_end, key_offset + k.shape[1])
        local_idx = torch.arange(local_abs_start, max(local_abs_start, local_abs_end), device=q.device) - key_offset
        key_idx = torch.cat((remote_idx, local_idx))
        # This includes the entire chosen remote set and enough local tokens
        # for every query row. The boolean mask enforces per-token causality.
        kb = kh.index_select(2, key_idx)
        vb = vh.index_select(2, key_idx)
        q_positions = torch.arange(q_abs_start, q_abs_end, device=q.device)
        k_positions = key_idx + key_offset
        allowed = (k_positions[None, :] <= q_positions[:, None]) & (
            (k_positions[None, :] >= q_positions[:, None] - local_window + 1)
            | (k_positions[None, :] < local_start)
        )
        # Bool SDPA mask uses True for keys participating in attention.
        y = F.scaled_dot_product_attention(
            qh[:, :, q_start:q_end], kb, vb,
            attn_mask=allowed[None, None, :, :], dropout_p=0.0, is_causal=False,
        )
        outputs.append(y)
        q_start = q_end
    return torch.cat(outputs, dim=2).transpose(1, 2).contiguous()


def flash_attn_qkvpacked_func(
    qkv: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim)
        causal: bool. Whether to apply causal attention mask.
    Return:
        out: (batch_size, seqlen, nheads, headdim).
    """
    return FlashAttnQKVPackedFunc.apply(
        qkv,
        softmax_scale,
        causal,
        torch.is_grad_enabled(),
    )


def flash_attn_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, headdim)
        kv: (batch_size, seqlen_k, 2, nheads_k, headdim)
        causal: bool. Whether to apply causal attention mask.
    Return:
        out: (batch_size, seqlen_q, nheads, headdim).
    """
    return FlashAttnKVPackedFunc.apply(
        q,
        kv,
        softmax_scale,
        causal,
        torch.is_grad_enabled(),
    )


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        causal: bool. Whether to apply causal attention mask.
    Return:
        out: (total_q, nheads, headdim).
    """
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        torch.is_grad_enabled(),
    )


def flash_attn_varlen_qkvpacked_func(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Arguments:
        qkv: (total, 3, nheads, headdim), where total = total number of tokens in the batch.
        cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
            of the sequences in the batch, used to index into qkv.
        max_seqlen: int. Maximum sequence length in the batch.
        causal: bool. Whether to apply causal attention mask.
    Return:
        out: (total, nheads, headdim).
    """
    return FlashAttnVarlenQKVPackedFunc.apply(
        qkv,
        cu_seqlens,
        max_seqlen,
        softmax_scale,
        causal,
        torch.is_grad_enabled(),
    )


def flash_attn_varlen_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        kv: (total_k, 2, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        causal: bool. Whether to apply causal attention mask.
    Return:
        out: (total_q, nheads, headdim).
    """
    return FlashAttnVarlenKVPackedFunc.apply(
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        torch.is_grad_enabled(),
    )
