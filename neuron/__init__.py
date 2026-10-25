"""Neuron: a small, fast LLM training stack aimed at a 1x/2x T4 budget.

Core modules:
  config        - typed YAML config + validation + budget math
  model         - pre-norm decoder-only transformer (RoPE, GQA, SwiGLU, tied embeddings)
  data          - memmapped uint16 token shards + deterministic stream sampler
  train         - AMP(fp16) loop: grad accum, Muon + AdamW_8bit, WSD schedule, ckpt/resume
  prepare_data  - HF datasets -> tokenized uint16 shards + manifest
  train_tokenizer- byte-level BPE tokenizer training
  bench         - throughput / memory benchmark + batch-size finder
  generate      - sampling from a checkpoint
  evaluate      - held-out loss/perplexity for a checkpoint
"""

__version__ = "0.1.0"
