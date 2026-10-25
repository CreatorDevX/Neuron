"""Typed configuration, validation, and token/compute budget math.

Configs are YAML files with optional `model:` and `train:` sections. Every field
has a default here so a config file only needs to override what it cares about.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import yaml


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12          # query heads
    n_kv_heads: int = 4        # key/value heads (GQA). Set == n_heads for MHA.
    ffn_hidden: int = 2048     # SwiGLU intermediate
    seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    init_std: float = 0.02
    attention: str = "efficient"  # efficient | math | flash | flash_turing | block_sparse | sparse
    hybrid_gdn: bool = False     # layer 0 then three GDN layers, repeating
    gdn_num_heads: int = 0       # 0 = n_heads
    gdn_head_dim: int = 0        # 0 = head_dim
    gdn_conv_kernel_size: int = 4
    gdn_backend: str = "auto"    # auto | fla | torch
    sparse_top_k: int = 2048      # candidate past keys per query when attention=sparse
    sparse_block_size: int = 128  # block size for attention=block_sparse
    sparse_top_blocks: int = 4    # evenly spaced remote blocks per query block
    local_window: int = 128       # exact causal sliding window in block_sparse mode
    use_engram: bool = False      # hashed n-gram memory with gated residual fusion
    pad_token_id: int = 1         # specials.py / tokenizer contract: EOD=0, PAD=1
    engram_ngram: int = 3         # N-gram order (2=bigram, 3=trigram)
    engram_vocab_size: tuple = (160000, 160000)  # Per-order vocab (NOT vocab_size**2!)
    engram_table_dim: int = 64   # embedding width per n-gram order, split across four hash heads
    loss_chunk: int = 128      # tokens per CE chunk (bounds logits memory)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def validate(self) -> None:
        if self.vocab_size < 2 or self.n_layers < 1 or self.seq_len < 1:
            raise ValueError("vocab_size, n_layers and seq_len must be positive (vocab_size >= 2)")
        if self.n_heads < 1 or self.n_kv_heads < 1:
            raise ValueError("n_heads and n_kv_heads must be >= 1")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) not divisible by n_heads ({self.n_heads})")
        if self.d_model % self.n_kv_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) not divisible by n_kv_heads ({self.n_kv_heads})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) not divisible by n_kv_heads ({self.n_kv_heads})")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if self.head_dim < 16:
            raise ValueError("head_dim < 16 is too small")
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id must be inside vocab_size")
        if self.ffn_hidden % 64 != 0 or self.d_model % 64 != 0:
            raise ValueError("d_model and ffn_hidden should be multiples of 64 for good GEMMs")
        if not (1 <= self.loss_chunk <= self.seq_len):
            raise ValueError(f"loss_chunk must be in [1, seq_len], got {self.loss_chunk}")
        if self.attention not in ("efficient", "math", "flash", "flash_turing", "block_sparse", "auto", "sparse"):
            raise ValueError(f"unknown attention backend {self.attention!r}")
        if self.gdn_num_heads < 0 or self.gdn_head_dim < 0:
            raise ValueError("gdn_num_heads and gdn_head_dim must be >= 0 (0 means derive from attention)")
        if self.gdn_conv_kernel_size < 1:
            raise ValueError("gdn_conv_kernel_size must be >= 1")
        if self.gdn_backend not in ("auto", "fla", "torch"):
            raise ValueError("gdn_backend must be one of: auto, fla, torch")
        gdn_heads = self.gdn_num_heads or self.n_heads
        gdn_dim = self.gdn_head_dim or self.head_dim
        if self.hybrid_gdn and (gdn_heads * gdn_dim < 1):
            raise ValueError("hybrid GDN requires positive head count and head dimension")
        if self.sparse_top_k < 1:
            raise ValueError("sparse_top_k must be >= 1")
        if self.attention == "block_sparse":
            if self.sparse_block_size < 16 or self.sparse_block_size % 16:
                raise ValueError("sparse_block_size must be a multiple of 16 and at least 16")
            if self.sparse_top_blocks < 0:
                raise ValueError("sparse_top_blocks must be >= 0")
            if not 1 <= self.local_window <= self.seq_len:
                raise ValueError("local_window must be in [1, seq_len]")
        if self.use_engram:
            if not 2 <= self.engram_ngram <= 8:
                raise ValueError("engram_ngram must be between 2 and 8")
            if not self.engram_vocab_size or any(int(v) < 2 for v in self.engram_vocab_size):
                raise ValueError("engram_vocab_size must contain positive table sizes >= 2")
            if self.engram_table_dim < 1:
                raise ValueError("engram_table_dim must be >= 1")


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    data_dir: str = "data"
    out_dir: str = "runs/run0"
    tokenizer_path: str = "tokenizer/tokenizer.json"  # used to check vocab + sample
    val_split: str = "val"

    # global batch = micro_batch * grad_accum * world_size sequences of seq_len
    micro_batch: int = 16
    grad_accum: int = 4

    # Either set total_tokens (recommended; steps are derived) or max_steps.
    total_tokens: int = 10_000_000_000
    train_on_all_data: bool = False  # replace total_tokens with manifest train token count
    max_steps: int = 0  # if > 0, overrides total_tokens

    lr: float = 8.0e-4
    min_lr_ratio: float = 0.05
    schedule: str = "wsd"       # wsd | cosine
    warmup_tokens: int = 100_000_000   # ~2000 steps at 48k tok/step (2 GPUs scale for free)
    warmup_steps: int = 0       # optional step-based warmup (overrides warmup_tokens if > 0)
    stable_ratio: float = 0.8   # WSD: fraction of post-warmup steps held flat
    decay_shape: str = "linear" # WSD decay shape: linear | cosine | sqrt
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    grad_clip_every: int = 1   # 1 preserves clipping every optimizer step; >1 is a speed/quality tradeoff
    optimizer_type: str = "muon_adamw_8bit"  # Muon for linear matrices, AdamW8bit otherwise

    dtype: str = "auto"         # auto | float16 | bfloat16 | float32
    compile: bool = True
    compile_mode: str = "default"          # default | max-autotune-no-cudagraphs | max-autotune
    compile_dynamic: bool = False
    grad_checkpointing: bool = False
    kv_chunk_size: int = 0       # optional differentiable cached training chunks; 0 = regular full sequence
    seed: int = 1337

    num_workers: int = 2
    prefetch_factor: int = 8
    pin_memory: bool = True
    log_every: int = 25
    eval_every: int = 2000           # steps
    eval_batch: int = 16             # sequences per eval micro-batch
    eval_batches: int = 24           # number of eval micro-batches (~tokens shown in log)
    ckpt_every_steps: int = 2000
    ckpt_every_min: float = 30.0     # also checkpoint on wall-clock
    keep_last: int = 3

    sample_every: int = 0            # steps; 0 disables
    sample_prompts: tuple = ()       # filled by yaml list

    wandb_project: str = ""
    allow_tf32: bool = False
    anomaly_debug: bool = False


# --------------------------------------------------------------------------- #
# Loading / overrides
# --------------------------------------------------------------------------- #
def _parse_scalar(s: str) -> Any:
    s = s.strip()
    try:
        return yaml.safe_load(s)
    except Exception:
        return s


def _apply_overrides(raw: dict, overrides) -> None:
    """overrides: list of 'a.b=c' strings. Dots descend into model/train."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        key, val = item.split("=", 1)
        parts = [p for p in key.split(".") if p]
        node = raw
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_scalar(val)


def build_configs(yaml_path: Optional[str], overrides=None) -> Tuple[ModelConfig, TrainConfig]:
    raw: dict = {}
    if yaml_path:
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    _apply_overrides(raw, overrides)

    def pick(cls, section: str):
        keys = {f.name for f in dataclasses.fields(cls)}
        got = raw.get(section, {}) or {}
        unknown = set(got) - keys
        if unknown:
            raise ValueError(f"unknown {section} keys: {sorted(unknown)} (known: {sorted(keys)})")
        return cls(**got)

    mcfg = pick(ModelConfig, "model")
    tcfg = pick(TrainConfig, "train")
    mcfg.validate()
    # cross validation
    if tcfg.micro_batch <= 0 or tcfg.grad_accum <= 0:
        raise ValueError("micro_batch and grad_accum must be > 0")
    if tcfg.grad_clip_every < 1:
        raise ValueError("grad_clip_every must be >= 1")
    if tcfg.schedule not in ("wsd", "cosine"):
        raise ValueError(f"unknown schedule {tcfg.schedule!r}")
    if tcfg.eval_batch < 1 or tcfg.eval_batches < 1:
        raise ValueError("eval_batch/eval_batches must be >= 1")
    if tcfg.log_every < 1 or tcfg.eval_every < 0 or tcfg.ckpt_every_steps < 0:
        raise ValueError("log_every must be positive; eval_every and ckpt_every_steps must be >= 0")
    if tcfg.ckpt_every_min < 0 or tcfg.keep_last < 1:
        raise ValueError("ckpt_every_min must be >= 0 and keep_last must be >= 1")
    if tcfg.num_workers < 0 or tcfg.prefetch_factor < 1:
        raise ValueError("num_workers must be >= 0 and prefetch_factor must be >= 1")
    if tcfg.kv_chunk_size < 0:
        raise ValueError("kv_chunk_size must be >= 0")
    if tcfg.kv_chunk_size and tcfg.compile:
        raise ValueError("kv_chunk_size requires train.compile=false")
    if tcfg.kv_chunk_size and tcfg.grad_checkpointing:
        raise ValueError("kv_chunk_size cannot be combined with gradient checkpointing")
    if tcfg.kv_chunk_size and mcfg.attention == "sparse":
        raise ValueError("kv_chunk_size is unsupported by the legacy experimental top-k backend")
    if tcfg.optimizer_type != "muon_adamw_8bit":
        raise ValueError("optimizer_type must be 'muon_adamw_8bit'")
    if tcfg.lr <= 0 or tcfg.weight_decay < 0:
        raise ValueError("lr must be > 0 and weight_decay must be >= 0")
    if not (0 <= tcfg.beta1 < 1 and 0 <= tcfg.beta2 < 1) or tcfg.eps <= 0:
        raise ValueError("AdamW betas must be in [0, 1) and eps must be > 0")
    if tcfg.grad_clip <= 0 or tcfg.warmup_tokens < 0 or tcfg.warmup_steps < 0:
        raise ValueError("grad_clip must be > 0 and warmup settings must be >= 0")
    if tcfg.total_tokens < 1 or tcfg.max_steps < 0:
        raise ValueError("total_tokens must be > 0 and max_steps must be >= 0")
    if tcfg.min_lr_ratio < 0 or tcfg.min_lr_ratio > 1:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    if not 0 <= tcfg.stable_ratio <= 1:
        raise ValueError("stable_ratio must be in [0, 1]")
    if tcfg.dtype not in ("auto", "float16", "bfloat16", "float32"):
        raise ValueError("dtype must be auto, float16, bfloat16, or float32")
    if tcfg.decay_shape not in ("linear", "cosine", "sqrt"):
        raise ValueError("decay_shape must be linear, cosine, or sqrt")
    return mcfg, tcfg


# --------------------------------------------------------------------------- #
# Budget math
# --------------------------------------------------------------------------- #
def count_params(mcfg: ModelConfig, separators: bool = True) -> int:
    """Analytic parameter count (matches `neuron.model.GPT` construction).

    Counts the shared token-embedding/lm-head matrix once (tied).
    """
    d = mcfg.d_model
    hd = mcfg.head_dim
    h = mcfg.n_heads
    kv = mcfg.n_kv_heads
    layer = (
        h * hd * d            # q_proj
        + kv * hd * d         # k_proj
        + kv * hd * d         # v_proj
        + h * hd * d          # out_proj
        + 3 * d * mcfg.ffn_hidden  # SwiGLU gate/up/down
        + 2 * d               # RMSNorm weights
    )
    engram_extra = 0
    if getattr(mcfg, 'use_engram', False):
        # Mirror Engram's four distinct prime-sized tables per n-gram order.
        requested = getattr(mcfg, 'engram_vocab_size', (160000, 160000))
        seen_primes = set()
        table_rows = 0
        table_count = 0

        def next_unused_prime(start: int) -> int:
            candidate = max(2, start)
            while True:
                prime = candidate >= 2 and all(
                    candidate % divisor for divisor in range(2, int(candidate ** 0.5) + 1)
                )
                if prime and candidate not in seen_primes:
                    return candidate
                candidate += 1

        for n in range(2, mcfg.engram_ngram + 1):
            idx = n - 2
            start = int(requested[idx]) if idx < len(requested) else 128000
            for _ in range(4):
                p = next_unused_prime(start)
                seen_primes.add(p)
                table_rows += p
                table_count += 1
        per_head = max(16, int(mcfg.engram_table_dim) // 4)
        embedded_dim = table_count * per_head
        engram_extra = table_rows * per_head + embedded_dim * d + 2 * d * d
    # The model attaches this memory module to the first four layers.
    engram_layers = min(4, mcfg.n_layers) if getattr(mcfg, 'use_engram', False) else 0
    total = mcfg.n_layers * layer + engram_layers * engram_extra + d  # final norm
    if mcfg.tie_embeddings:
        total += mcfg.vocab_size * d
    else:
        total += 2 * mcfg.vocab_size * d
    return total


def non_embedding_params(mcfg: ModelConfig) -> int:
    full = count_params(mcfg)
    if mcfg.tie_embeddings:
        return full - mcfg.vocab_size * mcfg.d_model
    return full - 2 * mcfg.vocab_size * mcfg.d_model


def world_tokens_per_step(tcfg: TrainConfig, world_size: int, mcfg: ModelConfig) -> int:
    """Tokens consumed per optimizer step across the whole cluster."""
    return tcfg.micro_batch * tcfg.grad_accum * world_size * mcfg.seq_len


def total_steps(tcfg: TrainConfig, world_size: int, mcfg: ModelConfig) -> int:
    if tcfg.max_steps > 0:
        return tcfg.max_steps
    tps = world_tokens_per_step(tcfg, world_size, mcfg)
    return max(1, math.ceil(tcfg.total_tokens / tps))


def warmup_steps_(tcfg: TrainConfig, world_size: int, mcfg: ModelConfig) -> int:
    steps = total_steps(tcfg, world_size, mcfg)
    if tcfg.warmup_steps > 0:
        return min(tcfg.warmup_steps, max(1, steps - 10))
    if tcfg.warmup_tokens > 0:
        tps = world_tokens_per_step(tcfg, world_size, mcfg)
        return max(1, min(steps - 10, int(math.ceil(tcfg.warmup_tokens / tps))))
    return 0


def approx_flops_per_token(mcfg: ModelConfig) -> float:
    """Approximate training FLOPs per token (forward+backward).

    Uses the 6*N convention for matmuls (N = all trainable params, embedding
    cost dominated by the tied lm_head matmul) plus a seq_len-dependent
    attention term. Good enough for MFU bookkeeping (±10%).
    """
    n = count_params(mcfg)  # exact: lm_head matmul is counted via N (see docstring)
    attn = 12.0 * mcfg.seq_len * mcfg.d_model * mcfg.n_layers
    return 6.0 * n + attn


def peak_flops(device_index: int = 0) -> float:
    """Rough fp16 tensor-core peak for MFU denominators.

    T4 (sm_75): 65.13 TFLOPS. Falls back to a conservative 40 TFLOPS.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 40e12
        p = torch.cuda.get_device_properties(device_index)
        if p.major == 7 and p.minor == 5:
            return 65.13e12
        return 40e12
    except Exception:
        return 40e12


def days_for_tokens(tokens: float, tok_per_s: float) -> float:
    return tokens / tok_per_s / 86400.0


def human_bytes(x: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(x) < 1024:
            return f"{x:.1f}{unit}"
        x /= 1024.0
    return f"{x:.1f}TB"


def human_int(x: float) -> str:
    return f"{x:,.0f}"
