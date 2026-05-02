"""Register vlm::fused_epilogue_add3_rmsnorm custom op.

Cross-layer epilogue fusion: combines layer N's post-MLP residual add
with layer N+1's input RMSNorm into a single opaque op.

    Layer N epilogue:   new_hidden = residual + mlp_out + DeepStack     (3-way add)
    Layer N+1 prologue: normed = RMSNorm(new_hidden)

Uses PyTorch native ops internally (fp32 variance + rsqrt) to guarantee
exact numerical match with eager. The custom op boundary prevents inductor
from decomposing RMSNorm into different accumulation order.
"""

from __future__ import annotations

import torch


@torch.library.custom_op("rldx_backbone::fused_epilogue_add3_rmsnorm", mutates_args=())
def fused_epilogue_add3_rmsnorm(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    deepStack: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-layer epilogue: 3-way residual add + RMSNorm.

    Args:
        hidden:   (B, M, D) bf16 — layer output (e.g. o_proj or mlp_out)
        residual: (B, M, D) bf16 — residual input (pre-attention or pre-MLP)
        deepStack: (B, M, D) bf16 — DeepStack visual features for this layer
        weight:   (D,) bf16 — next layer's input_layernorm weight (or final norm weight)

    Returns:
        new_hidden: (B, M, D) bf16 — hidden + residual + deepStack
        normed:     (B, M, D) bf16 — RMSNorm(new_hidden)
    """
    from ..kernels.fused_add3_rmsnorm import forward as _triton_forward

    return _triton_forward(hidden, residual, deepStack, weight)


@fused_epilogue_add3_rmsnorm.register_fake
def _(hidden, residual, deepStack, weight):
    return (torch.empty_like(hidden), torch.empty_like(hidden))
