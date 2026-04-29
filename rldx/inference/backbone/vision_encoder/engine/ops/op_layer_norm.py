"""Register vlm::layer_norm custom op.

Wraps F.layer_norm: opaque to torch.compile, preventing inductor from
decomposing LayerNorm into Welford reduction (which uses a different
accumulation order than PyTorch's native CUDA kernel, causing numerical diff).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.library.custom_op("rldx_backbone::layer_norm", mutates_args=())
def layer_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Native LayerNorm matching eager precision.

    Args:
        input:  (*batch, D) bf16
        weight: (D,) bf16 — LayerNorm weight
        bias:   (D,) bf16 — LayerNorm bias
        eps:    float — epsilon for numerical stability

    Returns:
        (*batch, D) bf16 — normalized output
    """
    return F.layer_norm(input, weight.shape, weight, bias, eps)


@layer_norm.register_fake
def _(input, weight, bias, eps):
    return torch.empty_like(input)
