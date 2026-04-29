"""Register vlm::fused_epilogue_add2_layernorm custom op.

Fuses 2-input residual add + LayerNorm into a single opaque op.
Naming follows the LLM convention (fused_epilogue_add2_rmsnorm).

Uses PyTorch native ops internally (bf16 add + F.layer_norm) to guarantee
exact numerical match with eager. The custom op boundary prevents inductor
from decomposing LayerNorm into Welford reduction (different accumulation order).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.library.custom_op("rldx_backbone::fused_epilogue_add2_layernorm", mutates_args=())
def fused_epilogue_add2_layernorm(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused residual add + LayerNorm.

    Args:
        hidden:   (M, D) bf16 — layer output (e.g. attention out)
        residual: (M, D) bf16 — residual input
        weight:   (D,) bf16 — LayerNorm weight
        bias:     (D,) bf16 — LayerNorm bias
        eps:      float — LayerNorm epsilon

    Returns:
        new_hidden: (M, D) bf16 — hidden + residual
        normed:     (M, D) bf16 — LayerNorm(new_hidden)
    """
    new_hidden = hidden + residual
    normed = F.layer_norm(new_hidden, weight.shape, weight, bias, eps)
    return new_hidden, normed


@fused_epilogue_add2_layernorm.register_fake
def _(hidden, residual, weight, bias, eps):
    return (torch.empty_like(hidden), torch.empty_like(hidden))
