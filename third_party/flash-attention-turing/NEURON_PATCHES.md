# Neuron changes

This tree is based on upstream [ssiu/flash-attention-turing](https://github.com/ssiu/flash-attention-turing),
commit `9ef98fcb506bb1e2fe3cece50935e2935bf6b124`.

Neuron adds `flash_attn_block_sparse_func` to `flash_attention_interface.py` and
packages the helper from `setup.py`. It computes causal attention over selected
remote KV blocks plus a local sliding window using PyTorch SDPA over compacted
keys. The helper supports cached Q/K/V ranges at absolute positions. It is a
reference path with Python/gather overhead, not a fused CUDA block-sparse
kernel. The upstream SM75 CUDA kernels are unchanged.

The upstream repository currently has no root LICENSE file. See the root
`THIRD_PARTY_NOTICES.md`; Neuron's Apache-2.0 license does not apply to the
third-party source.
