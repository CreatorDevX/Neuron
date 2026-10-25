"""Reuse the configured tokenizer or optionally train a byte-level BPE tokenizer.

Default (Llama 2 tokenizer from the data spec):
  python -m neuron.train_tokenizer --spec configs/data_spec.yaml --out tokenizer

Optional new tokenizer:
  python -m neuron.train_tokenizer --spec configs/data_spec.yaml --train-new --out tokenizer

Output:
  tokenizer/tokenizer.json
  tokenizer/tokenizer.meta.json   (vocab_size, specials ids)

For `--train-new`, the tokenizer is a 32k byte-level BPE trained from a corpus
sample, with Neuron's reserved special tokens (EOD=0, PAD=1).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from .specials import EOD, PAD, reserved


def sample_docs(spec: dict, per_source_mb: float, out_path: Path, seed: int) -> int:
    """Write up to per_source_mb MB of text per source into out_path."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from .prepare_data import iter_examples

    total = 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for src in spec.get("sources", []):
            sid = src.get("id", "src")
            cap = int(per_source_mb * 1024 * 1024)
            written = 0
            for i, text in enumerate(iter_examples(src, seed=seed, src_index=0)):
                if not text:
                    continue
                text = text[:200_000]
                enc_len = len(text.encode("utf-8"))
                f.write(text.replace("\r\n", "\n") + "\n")
                written += enc_len
                if written >= cap:
                    print(f"  [{sid}] sampled {written/1024/1024:.0f}MB")
                    break
            total += written
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="configs/data_spec.yaml")
    ap.add_argument("--out", default="tokenizer")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--sample-mb", type=float, default=150.0)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--pretrained", default=None,
                    help="use this pretrained tokenizer repo instead of the one in the spec")
    ap.add_argument("--train-new", action="store_true",
                    help="train a new byte-level BPE tokenizer instead of reusing a pretrained one")
    args = ap.parse_args()

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    spec = {}
    with open(args.spec, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f) or {}
    if args.train_new and args.pretrained:
        ap.error("--train-new and --pretrained cannot be used together")
    pretrained_repo = None if args.train_new else (args.pretrained or spec.get("tokenizer_repo"))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_path = out_dir / "tokenizer.json"
    if pretrained_repo:
        from transformers import AutoTokenizer

        hf_token = os.environ.get("HF_TOKEN") or None
        print(f"[tokenizer] loading {pretrained_repo} (requires accepted HF model terms)", flush=True)
        pretrained = AutoTokenizer.from_pretrained(
            pretrained_repo, token=hf_token, use_fast=True,
        )
        backend = getattr(pretrained, "backend_tokenizer", None)
        if backend is None:
            raise RuntimeError("the selected tokenizer has no fast/tokenizers backend")
        backend.save(str(tok_path))
        meta = {
            "vocab_size": backend.get_vocab_size(with_added_tokens=True),
            "eod_id": pretrained.eos_token_id,
            "bos_id": pretrained.bos_token_id,
            "pad_id": pretrained.pad_token_id,
            "kind": "pretrained",
            "repo": pretrained_repo,
        }
        if meta["eod_id"] is None:
            raise ValueError("selected tokenizer has no EOS token")
        with open(out_dir / "tokenizer.meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[tokenizer] vocab={meta['vocab_size']} eos_id={meta['eod_id']} -> {tok_path}", flush=True)
        return

    sample_file = out_dir / "corpus.sample.txt"
    print(f"[tokenizer] sampling corpus to {sample_file} ...")
    n = sample_docs(spec, args.sample_mb, sample_file, args.seed)
    print(f"[tokenizer] sampled {n/1024/1024:.0f}MB total")

    specials = [EOD, PAD] + reserved(14)  # 16 total; EOD id=0, PAD id=1

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=specials,
        min_frequency=args.min_freq,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    print(f"[tokenizer] training {args.vocab_size}-vocab BPE ...")
    tokenizer.train([str(sample_file)], trainer=trainer)
    tokenizer.no_padding(), tokenizer.no_truncation()

    tokenizer.save(str(tok_path))
    eod_id = tokenizer.token_to_id(EOD)
    pad_id = tokenizer.token_to_id(PAD)
    assert eod_id is not None and eod_id == 0, f"EOD id {eod_id} (want 0)"
    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "eod_id": eod_id,
        "pad_id": pad_id,
        "specials": specials,
        "kind": "byte-bpe",
    }
    with open(out_dir / "tokenizer.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # smoke: encode/decode roundtrip
    ex = "The quick brown fox jumps over the lazy dog. Numbers: 12,345 and 0.125!"
    ids = tokenizer.encode(ex).ids
    rt = tokenizer.decode(ids, skip_special_tokens=False)
    print(f"[tokenizer] vocab={tokenizer.get_vocab_size()} EOD={eod_id} PAD={pad_id}")
    print(f"[tokenizer] roundtrip ok={rt == ex}")
    if tokenizer.get_vocab_size() != args.vocab_size:
        print(f"[tokenizer][warn] final vocab is {tokenizer.get_vocab_size()} "
              f"(settings said {args.vocab_size}); manifest will use the real number")
    print(f"[tokenizer] wrote {tok_path}")


if __name__ == "__main__":
    sys.exit(main())
