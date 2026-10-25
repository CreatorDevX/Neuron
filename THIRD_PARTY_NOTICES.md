# Third-party notices

## FlashAttention for Turing GPUs

The source in `third_party/flash-attention-turing/` is based on
[ssiu/flash-attention-turing](https://github.com/ssiu/flash-attention-turing),
upstream commit `9ef98fcb506bb1e2fe3cece50935e2935bf6b124`.

Neuron's local changes add `flash_attn_block_sparse_func` to the Python
interface. It implements selected remote key/value blocks plus a causal local
window using PyTorch SDPA, and accepts absolute query/key offsets for cached
attention. This is a reference implementation, not a fused sparse CUDA kernel.
Neuron's bounded sliding-window replay logic is implemented in `neuron/model.py`.

The upstream repository currently has no `LICENSE` file in its root listing.
Neuron's Apache-2.0 license does not apply to this third-party source and does
not grant rights on behalf of its authors. Its rights and any applicable terms
remain with the upstream copyright holders. This copy is retained temporarily
for benchmarking while licensing is clarified; this notice does not grant a
license or permission to redistribute the upstream code. See the upstream
project for its current status before using or redistributing that source.

The Turing CUDA extension is based on the upstream project; the SDPA helper
described above is Neuron's local modification. CUTLASS is fetched separately
at `df18f5e4f5de76bed8be1de8e4c245f2f5ec3020` as declared by the Turing
project. CUTLASS is BSD-3-Clause licensed; retain its `LICENSE.txt` notices
when redistributing it.
