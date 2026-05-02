"""Register vlm::vision_attention custom op.

Fused vision RoPE + non-causal attention with Split-KV (Flash-Decoding).
Reads Q/K/V directly from fused QKV buffer (no permute/copy).
RoPE sign from rotate_half is pre-baked into rope_sin.
NUM_SPLITS is chosen at kernel launch via @triton.autotune (no caller config).
"""

from __future__ import annotations

import torch


@torch.library.custom_op("rldx_backbone::vision_attention", mutates_args=())
def vision_attention(
    qkv: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scaling: float,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Fused vision RoPE + non-causal attention (baked cos/sin, fused QKV).

    Args:
        qkv:        (M, 3 * num_heads * head_dim) bf16 — fused QKV
        rope_cos:   (M, head_dim) — baked cos (static)
        rope_sin:   (M, head_dim) — baked sin with rotate_half sign (static)
        cu_seqlens: (num_seqs+1,) int32
        scaling:    float — head_dim ** -0.5
        num_heads:  int
        head_dim:   int

    Returns:
        (M, num_heads * head_dim) bf16
    """
    from backbone.vision_encoder.engine.kernels.fused_vision_attention import forward

    return forward(qkv, rope_cos, rope_sin, cu_seqlens, scaling, num_heads, head_dim)


@vision_attention.register_fake
def _(qkv, rope_cos, rope_sin, cu_seqlens, scaling, num_heads, head_dim):
    M = qkv.shape[0]
    return qkv.new_empty((M, num_heads * head_dim))
