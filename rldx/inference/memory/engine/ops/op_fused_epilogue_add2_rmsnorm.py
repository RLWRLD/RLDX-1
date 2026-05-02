"""Register mem::fused_epilogue_add2_rmsnorm custom op.

Cross-layer epilogue fusion: residual add + next layer's RMSNorm.
Reuses VLM's Triton kernel (generic, parameterized by D).

Dtype (matching eager RMSNorm exactly):
  new_hidden = hidden + residual  (fp32 add, bf16 store)
  normed = weight * (new_hidden * rsqrt(var + eps)).to(bf16)  — cast before weight
"""

from __future__ import annotations

import torch


@torch.library.custom_op("mem::fused_epilogue_add2_rmsnorm", mutates_args=())
def fused_epilogue_add2_rmsnorm(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-layer epilogue: 2-way residual add + RMSNorm.

    Args:
        hidden:   (B, M, D) bf16 — layer output (attn_out or mlp_out)
        residual: (B, M, D) bf16 — skip connection
        weight:   (D,) bf16 — next layer's input_layernorm weight

    Returns:
        new_hidden: (B, M, D) bf16 — hidden + residual
        normed:     (B, M, D) bf16 — RMSNorm(new_hidden)
    """
    from memory.engine.kernels.fused_add2_rmsnorm import forward

    return forward(hidden, residual, weight)


@fused_epilogue_add2_rmsnorm.register_fake
def _(hidden, residual, weight):
    return (torch.empty_like(hidden), torch.empty_like(hidden))
