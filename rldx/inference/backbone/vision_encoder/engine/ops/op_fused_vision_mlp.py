"""Register vlm::fused_vision_mlp custom op.

Wraps Vision Encoder MLP (fc1 → GELU → fc2) as a single opaque op.
Prevents inductor from decomposing addmm → mm + bias_add when fusing
Linear + activation, which truncates mm output to bf16 before bias addition
(eager addmm adds bias in internal f32 before the final bf16 cast).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.library.custom_op("rldx_backbone::fused_vision_mlp", mutates_args=())
def fused_vision_mlp(
    hidden_states: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
) -> torch.Tensor:
    """Vision encoder MLP matching eager dtype flow.

    Args:
        hidden_states: (M, hidden_size) bf16
        fc1_weight:    (intermediate_size, hidden_size) bf16
        fc1_bias:      (intermediate_size,) bf16
        fc2_weight:    (hidden_size, intermediate_size) bf16
        fc2_bias:      (hidden_size,) bf16

    Returns:
        (M, hidden_size) bf16
    """
    # F.linear uses addmm for 2D input → bias added in cuBLAS internal f32
    h = F.linear(hidden_states, fc1_weight, fc1_bias)
    h = F.gelu(h, approximate="tanh")
    return F.linear(h, fc2_weight, fc2_bias)


@fused_vision_mlp.register_fake
def _(hidden_states, fc1_weight, fc1_bias, fc2_weight, fc2_bias):
    M = hidden_states.shape[0]
    out_features = fc2_weight.shape[0]
    return hidden_states.new_empty((M, out_features))
