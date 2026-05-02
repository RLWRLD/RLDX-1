"""Triton kernels for TransformerMemory layers.

- fused_memory_attention: RoPE + block-causal SDPA
- fused_add2_rmsnorm: cross-layer residual add + RMSNorm (reuses VLM kernel)
"""
