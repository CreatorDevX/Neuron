# Roadmap

Neuron is being developed as a complete workflow for compact language models.

## Available now

- Decoder-only pretraining with the Wikipedia data pipeline.
- Llama 2 tokenizer reuse by default, with optional byte-level BPE tokenizer training.
- Hybrid block-sparse attention, Gated DeltaNet layers, and Engram n-gram memory.
- Mixed-precision training, resumable checkpoints, held-out evaluation, and text generation.
- Colab/Kaggle profiles, throughput checks, and synthetic option ablations.

## Planned next

- Supervised fine-tuning (SFT): instruction data loading, supervised objectives, and resumable post-training runs.
- Preference and reinforcement-learning (RL) stages: preference data formats, reward/policy objectives, and evaluation hooks.
- A documented end-to-end workflow that connects pretraining, SFT, RL, evaluation, and checkpoint export.

These items describe the intended direction; SFT and RL trainers are not included in the current code.
