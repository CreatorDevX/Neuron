"""Generate text from a Neuron checkpoint.

  python -m neuron.generate --ckpt runs/neuron-100m/final.pt --prompt "The " \
      --max-new-tokens 200 --temperature 0.8 --top-k 50
"""
from __future__ import annotations

import argparse
import contextlib
import math
import os

import torch

from .config import ModelConfig
from .checkpoint import sha256_file
from .model import GPT
from .specials import EOD


def load(path: str, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    mcfg = ModelConfig(**state["model_cfg"])
    model = GPT(mcfg)
    model.load_state_dict(state["model"], strict=True)
    model.to(device).to(torch.bfloat16 if torch.cuda.is_available() and
                        torch.cuda.is_bf16_supported() else (torch.float16 if device.type == "cuda" else torch.float32))
    model.eval()
    return model, mcfg, state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=3, help="samples")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument(
        "--swa-bounded-replay",
        action="store_true",
        help="bound attention KV storage during replay and incremental decode",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mcfg, state = load(args.ckpt, device)
    if args.fp32:
        model = model.float()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    if tok.get_vocab_size() != mcfg.vocab_size:
        raise SystemExit(f"tokenizer vocab {tok.get_vocab_size()} != checkpoint vocab {mcfg.vocab_size}")
    expected_tokenizer = (state.get("identity") or {}).get("tokenizer_sha256")
    if expected_tokenizer and sha256_file(args.tokenizer) != expected_tokenizer:
        raise SystemExit("tokenizer hash differs from the checkpoint's training tokenizer")
    bos = tok.token_to_id("<s>")
    if bos is None:
        bos = tok.token_to_id(EOD)
    if bos is None:
        bos = tok.token_to_id("</s>")
    if bos is None:
        raise SystemExit(f"tokenizer is missing BOS token '<s>' or end token {EOD}/'</s>'")
    torch.manual_seed(args.seed)

    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device.type == "cuda" and not args.fp32 else contextlib.nullcontext())
    ids = tok.encode(args.prompt, add_special_tokens=False).ids
    idx = torch.tensor([[bos] + ids], dtype=torch.long, device=device)

    with ctx, torch.no_grad():
        for s in range(args.n):
            torch.manual_seed(args.seed + s)
            out = model.generate(idx, args.max_new_tokens, temperature=args.temperature,
                                 top_k=args.top_k, eos_id=tok.token_to_id("</s>") or
                                 tok.token_to_id(EOD),
                                 swa_bounded_replay=args.swa_bounded_replay)
            text = tok.decode(out[0, idx.shape[1]:].tolist(), skip_special_tokens=False)
            print(f"--- sample {s} --------------------------------------------------")
            print(text)
            print()


if __name__ == "__main__":
    main()
