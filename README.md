# Neuron

Neuron is a public project for training and evaluating compact decoder-only language models on NVIDIA T4 GPUs. The repository includes the model, Wikipedia data pipeline, tokenizer setup, trainer, resumable checkpoints, evaluation, generation, benchmarks, and a Google Colab notebook.

The default preset combines block-sparse causal attention, grouped-query attention, a repeating attention/Gated DeltaNet (GDN) layer pattern, and Engram hashed n-gram memory. It is approximately 146M parameters with the current Engram tables. The block-sparse path is a PyTorch SDPA reference implementation; it does not require a separate attention fork. GDN uses Flash Linear Attention when available and can fall back to its PyTorch reference implementation.

## Current status and roadmap

Pretraining is implemented. The project is being developed as an end-to-end language-model training toolkit. Supervised fine-tuning (SFT) and reinforcement-learning (RL) training are planned next, with the goal of shipping the full pretraining-to-post-training workflow in this repository. Those SFT and RL trainers are not in the current release yet; see the [roadmap](ROADMAP.md).

## Quick start

The supported training environment is Linux with a CUDA-enabled PyTorch installation. The main recipe targets NVIDIA T4 GPUs. For a guided setup, open [`notebooks/neuron-wikipedia-colab.ipynb`](notebooks/neuron-wikipedia-colab.ipynb) in Google Colab, select a GPU runtime, and follow the cells in order.

For a local or hosted Linux GPU environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install "flash-linear-attention[cuda]"
```

Copy `.env.example` to `.env` and fill in credentials for the services you use. The file is ignored by Git. Then launch the default preset:

```bash
cp .env.example .env
python -m neuron.train
```

Training automatically creates the configured tokenizer and prepares the dataset when those files are missing. To run a short real-data throughput check after preparation:

```bash
python -m neuron.pipeline_test --config configs/neuron-100m-t4.yaml --steps 3 \
  --set train.micro_batch=1 train.grad_accum=1 train.compile=false train.grad_checkpointing=true
```

The default full run follows the train-token count in the generated manifest. Use `--max-steps N` for a bounded run. Typed config fields can be overridden without editing YAML:

```bash
python -m neuron.train --config configs/neuron-100m-t4.yaml \
  --set train.micro_batch=2 train.grad_accum=32 model.local_window=128
```

## Tokenizer

The default is the fast tokenizer from [`meta-llama/Llama-2-7b-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf). Neuron downloads tokenizer assets only; it does not download or use Llama model weights. The Llama repository is gated: accept Meta's license terms and request access through Hugging Face before running setup. Set `HF_TOKEN` in `.env` to a Hugging Face user access token with read permission for the tokenizer.

To prepare the tokenizer explicitly:

```bash
python -m neuron.train_tokenizer --spec configs/data_spec.yaml --out tokenizer
```

Training a new byte-level BPE tokenizer is an optional alternative:

```bash
python -m neuron.train_tokenizer --spec configs/data_spec.yaml --train-new --out tokenizer
```

When starting a fresh run, `python -m neuron.train --train-new-tokenizer` selects
that option during automatic data setup. It requires a fresh data directory.

The model preset uses a 32,000-token vocabulary. When selecting a different tokenizer, keep its vocabulary size and special-token IDs consistent with the model and prepare a fresh data directory so existing shards are not reused with a different tokenizer.

## Data

The default data specification streams the public English Wikipedia dataset [`oriental-lab/wikipedia-english`](https://huggingface.co/datasets/oriental-lab/wikipedia-english). The pipeline tokenizes full non-empty articles, holds out a small deterministic validation sample, and writes resumable `uint16` shards plus a manifest. The prepared manifest determines the default training token budget.

Data preparation can take a long time and requires disk space for the token shards. To prepare manually:

```bash
python -m neuron.train_tokenizer --spec configs/data_spec.yaml --out tokenizer
python -m neuron.prepare_data --spec configs/data_spec.yaml --out data
```

The preparation cursor allows interrupted runs to resume. For longer Colab sessions, configure a Hugging Face Storage Bucket to sync data and checkpoints.

## Credentials

All credentials are optional unless you use the related service:

- `HF_TOKEN`: Hugging Face user access token. Read access is needed for the gated Llama 2 tokenizer; write access is needed to upload to a Hugging Face Storage Bucket.
- `HF_USERNAME` or `HF_NAMESPACE`: Hugging Face account or organization that owns the bucket.
- `HF_BUCKET_NAME` or `HF_BUCKET_ID`: name of the existing Storage Bucket, or its full `owner/name` id.
- `HF_BUCKET_PREFIX`: folder prefix within the bucket for Neuron files.
- `WANDB_API_KEY`: W&B API key for experiment tracking. `WANDB_ENTITY` is the user/team and `WANDB_PROJECT` is the project slug. Leave `WANDB_RUN_ID` blank to create a run automatically.

See [`.env.example`](.env.example) for the complete list and comments. Never commit `.env` or paste real keys into a public notebook.

## Project map

- `neuron/model.py` — transformer, GQA/RoPE, block-sparse attention, GDN, Engram, and KV cache.
- `neuron/train.py`, `neuron/optim.py` — training loop, mixed precision, optimizer, scheduling, and notebook launcher.
- `neuron/prepare_data.py`, `neuron/data.py` — streaming dataset preparation, shard manifests, and deterministic sampling.
- `neuron/train_tokenizer.py` — load the configured pretrained tokenizer or optionally train byte-level BPE.
- `neuron/checkpoint.py`, `neuron/hub.py` — resumable checkpoint metadata and optional Hugging Face Bucket synchronization.
- `neuron/evaluate.py`, `neuron/generate.py` — held-out loss/perplexity and checkpoint text generation.
- `neuron/bench.py`, `neuron/ablate.py`, `neuron/pipeline_test.py` — performance and implementation checks.
- `configs/` — main, Colab, Kaggle, smoke-test, and experimental configurations.
- `tests/` — numerical and CPU end-to-end smoke coverage.

## Turing FlashAttention benchmark dependency

The optional `flash_turing` backend is included under
`third_party/flash-attention-turing/` for T4 kernel benchmarking. Its custom
block-sparse helper supports routed remote KV blocks and absolute-position
offsets. Neuron's bounded sliding-window replay is implemented in `model.py`
and uses the model's offset-aware block-sparse path.
The normal block-sparse preset uses Neuron's own PyTorch reference path and
does not need this compiled extension.

Build the Turing extension from the repository root on a CUDA 12.x/T4 Linux
environment:

```bash
cd third_party/flash-attention-turing
if [ ! -d csrc/cutlass ]; then git clone https://github.com/NVIDIA/cutlass.git csrc/cutlass; fi
git -C csrc/cutlass checkout df18f5e4f5de76bed8be1de8e4c245f2f5ec3020
MAX_JOBS=2 python -m pip install -v .
cd ../..
python -m neuron.ablate --config configs/neuron-100m-t4.yaml --device cuda \\
  --sweep attention --attention flash_turing --micro-batch 1 --iters 1 --warmup 0
```

The extension source is based on the upstream project at its base commit; its
CUDA kernels are unchanged. CUTLASS is fetched separately at the pinned
submodule revision because it is large. Neuron's Python helper changes are documented in
[`third_party/flash-attention-turing/NEURON_PATCHES.md`](third_party/flash-attention-turing/NEURON_PATCHES.md).
The upstream repository currently has no root `LICENSE` file. The project
Apache-2.0 license does not cover these third-party files; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the current status.

## Configurations

- `configs/neuron-100m-t4.yaml` — default hybrid preset (about 146M parameters with Engram enabled).
- `configs/colab-1xt4.yaml` — single-GPU Colab profile.
- `configs/kaggle-2xt4.yaml` — two-GPU Kaggle profile.
- `configs/smoke-test.yaml` — small CPU workflow for development.
- `configs/neuron-32k-hybrid.yaml` — experimental long-context configuration.

Block-sparse attention is implemented in Neuron using PyTorch SDPA over selected remote blocks and a local causal window. This reference path is compatible with the configured layer pattern and KV cache, but has Python and gather overhead. The optional `flash_turing` backend requires the separately installed Turing extension; it is not required by the default preset.

## Tests and checks

```bash
python tests/test_numerics.py
python tests/e2e_smoke.py
```

A short real-data throughput check is available through `neuron.pipeline_test`. Synthetic kernel and option comparisons are available through `neuron.ablate`; these measure implementation performance, not model quality.

## License

Neuron's original source code is licensed under Apache-2.0; see [`LICENSE`](LICENSE).
Third-party components retain their own terms and are listed separately in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). In particular, the included
Turing attention source has no upstream license file at this time and is not
relicensed under Apache-2.0.
