"""Register mem::fused_attention custom op.

Fused RoPE + Block-Causal Attention for TransformerMemory.
No QK RMSNorm, no GQA. All RoPE ops in bf16 matching eager.
"""

from __future__ import annotations

import torch


@torch.library.custom_op("mem::fused_attention", mutates_args=())
def fused_memory_attention(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    signed_sin: torch.Tensor,
    num_heads: int,
    head_dim: int,
    block_attn_size: int,
) -> torch.Tensor:
    """RoPE + Block-Causal SDPA (Triton kernel).

    Args:
        qkv:             (M, QKV_DIM) bf16 — fused QKV projection output
        cos:             (M, D) bf16 — pre-computed RoPE cos
        signed_sin:      (M, D) bf16 — pre-computed signed sin
        num_heads:       int (16)
        head_dim:        int (256)
        block_attn_size: int (16)

    Returns:
        (M, num_heads * head_dim) bf16 — attention output (before o_proj)
    """
    from memory.engine.kernels.fused_memory_attention import forward

    return forward(qkv, cos, signed_sin, num_heads, head_dim, block_attn_size)


@fused_memory_attention.register_fake
def _(qkv, cos, signed_sin, num_heads, head_dim, block_attn_size):
    M = qkv.shape[0]
    return qkv.new_empty((M, num_heads * head_dim))
