"""Register vlm::fused_epilogue_add2_rmsnorm custom op.

Cross-layer epilogue fusion: combines layer N's post-MLP residual add
with layer N+1's input RMSNorm into a single opaque op.

    Layer N epilogue:   new_hidden = residual + mlp_out      (2-way add)
    Layer N+1 prologue: normed = RMSNorm(new_hidden)

Uses PyTorch native ops internally (fp32 variance + rsqrt) to guarantee
exact numerical match with eager. The custom op boundary prevents inductor
from decomposing RMSNorm into different accumulation order.
"""

from __future__ import annotations

import torch


@torch.library.custom_op("rldx_backbone::fused_epilogue_add2_rmsnorm", mutates_args=())
def fused_epilogue_add2_rmsnorm(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-layer epilogue: 2-way residual add + RMSNorm.

    Args:
        hidden:   (B, M, D) bf16 — layer output (e.g. o_proj or mlp_out)
        residual: (B, M, D) bf16 — residual input (pre-attention or pre-MLP)
        weight:   (D,) bf16 — next layer's input_layernorm weight (or final norm weight)

    Returns:
        new_hidden: (B, M, D) bf16 — hidden + residual
        normed:     (B, M, D) bf16 — RMSNorm(new_hidden)
    """
    from ..kernels.fused_add2_rmsnorm import forward as _triton_forward

    return _triton_forward(hidden, residual, weight)


@fused_epilogue_add2_rmsnorm.register_fake
def _(hidden, residual, weight):
    return (torch.empty_like(hidden), torch.empty_like(hidden))
