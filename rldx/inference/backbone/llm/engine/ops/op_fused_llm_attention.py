"""Register vlm::fused_attention custom op.

Fused RMSNorm + weight + RoPE + Causal Attention.
Norm weights stored per-layer; cos/sin shared per-chain.
Matches eager computation order exactly (bf16 × bf16 at each step).
"""

from __future__ import annotations

import torch


@torch.library.custom_op("rldx_backbone::fused_attention", mutates_args=())
def fused_llm_attention(
    qkv: torch.Tensor,
    q_norm_w: torch.Tensor,
    q_norm_w_rot: torch.Tensor,
    k_norm_w: torch.Tensor,
    k_norm_w_rot: torch.Tensor,
    cos: torch.Tensor,
    signed_sin: torch.Tensor,
) -> torch.Tensor:
    """RMSNorm + weight + RoPE + Causal Attention (Triton kernel).

    Args:
        qkv:            (M, QKV_DIM) bf16
        q_norm_w:       (128,) bf16 — per-layer Q norm weight
        q_norm_w_rot:   (128,) bf16 — per-layer Q norm weight (rotated)
        k_norm_w:       (128,) bf16 — per-layer K norm weight
        k_norm_w_rot:   (128,) bf16 — per-layer K norm weight (rotated)
        cos:            (M, 128) bf16 — shared RoPE cos
        signed_sin:     (M, 128) bf16 — shared signed sin

    Returns:
        (M, 4096) bf16 — attention output (before o_proj)
    """
    from backbone.llm.engine.kernels.fused_llm_attention import forward

    return forward(qkv, q_norm_w, q_norm_w_rot, k_norm_w, k_norm_w_rot, cos, signed_sin)


@fused_llm_attention.register_fake
def _(qkv, q_norm_w, q_norm_w_rot, k_norm_w, k_norm_w_rot, cos, signed_sin):
    M = qkv.shape[0]
    return qkv.new_empty((M, 4096))
