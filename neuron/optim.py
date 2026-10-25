"""Hybrid Muon + AdamW8bit optimizer and learning-rate schedules."""
from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn

from .config import TrainConfig


def split_muon_adamw_params(model: nn.Module):
    """Send linear matrix weights to Muon; embeddings, norms and vectors to AdamW."""
    muon_params, adamw_decay, adamw_no_decay = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_embedding = "emb" in name.lower()
        if param.ndim == 2 and name.endswith(".weight") and not is_embedding:
            muon_params.append(param)
        elif param.ndim < 2 or name.endswith("bias") or "norm" in name.lower():
            adamw_no_decay.append(param)
        else:
            adamw_decay.append(param)
    return muon_params, adamw_decay, adamw_no_decay


def build_optimizer(model: nn.Module, tcfg: TrainConfig, device: str):
    """Build the declared Muon + AdamW8bit optimizer, failing clearly if unavailable."""
    if tcfg.optimizer_type != "muon_adamw_8bit":
        raise ValueError(f"unsupported optimizer_type {tcfg.optimizer_type!r}")
    muon_cls = getattr(torch.optim, "Muon", None)
    if muon_cls is None:
        raise RuntimeError("Muon requires PyTorch >= 2.9; install the pinned runtime from requirements.txt")
    muon_params, adamw_decay, adamw_no_decay = split_muon_adamw_params(model)
    if not muon_params or not adamw_no_decay:
        raise ValueError("hybrid optimizer received an empty Muon or AdamW parameter partition")

    muon_opt = muon_cls(
        [{"params": muon_params, "weight_decay": float(tcfg.weight_decay)}],
        lr=float(tcfg.lr), momentum=0.95, nesterov=True, ns_steps=5,
    )
    adamw_groups = [
        {"params": adamw_decay, "weight_decay": float(tcfg.weight_decay)},
        {"params": adamw_no_decay, "weight_decay": 0.0},
    ]
    if not device.startswith("cuda"):
        # CPU smoke runs only validate the loop/model; the actual CUDA recipe is hybrid.
        warnings.warn("CPU smoke run uses full-precision AdamW; CUDA training uses AdamW8bit", RuntimeWarning)
        adamw_opt = torch.optim.AdamW(
            adamw_groups, lr=float(tcfg.lr), betas=(tcfg.beta1, tcfg.beta2), eps=tcfg.eps,
            foreach=True,
        )
    else:
        try:
            import bitsandbytes as bnb
        except Exception as exc:
            raise RuntimeError("CUDA training requires bitsandbytes for AdamW8bit; install requirements.txt") from exc
        adamw_opt = bnb.optim.AdamW8bit(
            adamw_groups, lr=float(tcfg.lr), betas=(tcfg.beta1, tcfg.beta2), eps=tcfg.eps,
        )
    return HybridOptimizer(muon_opt, adamw_opt)


class HybridOptimizer:
    """Small adapter that exposes both optimizers as one optimizer to the loop/scaler."""
    def __init__(self, muon_opt, adamw_opt):
        self.muon = muon_opt
        self.adamw = adamw_opt

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adamw.param_groups

    def step(self, closure=None):
        if closure is not None:
            raise ValueError("HybridOptimizer does not support closures")
        self.muon.step()
        self.adamw.step()

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state_dict):
        if "muon" not in state_dict or "adamw" not in state_dict:
            raise ValueError("checkpoint does not contain both hybrid optimizer states")
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])


def lr_at(step: int, tcfg: TrainConfig, steps_total: int, warmup: int) -> float:
    """LR for a step (0-based). Pure function so schedulers are audit-able."""
    peak = float(tcfg.lr)
    floor = peak * float(tcfg.min_lr_ratio)
    s = int(step)
    if steps_total <= warmup + 2:
        return peak

    def decay_linear(p):
        return floor + (peak - floor) * (1.0 - p)

    def decay_cosine(p):
        return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * min(1.0, p)))

    def decay_sqrt(p):
        return floor + (peak - floor) * (1.0 - math.sqrt(min(1.0, p)))

    if warmup > 0 and s < warmup:
        return peak * (s / max(1, warmup))
    if tcfg.schedule == "cosine":
        return decay_cosine((s - warmup) / max(1, steps_total - warmup))
    stable_end = warmup + int(math.floor(tcfg.stable_ratio * (steps_total - warmup)))
    if s < stable_end:
        return peak
    p = (s - stable_end) / max(1, steps_total - stable_end)
    if tcfg.decay_shape == "cosine":
        return decay_cosine(p)
    if tcfg.decay_shape == "sqrt":
        return decay_sqrt(p)
    return decay_linear(p)


def make_scaler(tcfg: TrainConfig, device: str, dtype: torch.dtype):
    if device.startswith("cuda") and dtype == torch.float16:
        return torch.amp.GradScaler(
            "cuda", init_scale=2**16, growth_factor=2.0,
            backoff_factor=0.5, growth_interval=2000, enabled=True,
        )
    return None
