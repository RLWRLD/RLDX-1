"""Shared primitive layers used across multiple modules (msat, memory, etc.)."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Shared by action head (MSAT) and memory module.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        x_normed = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x_normed).to(input_dtype)


def create_norm_layer(norm_type: str, dim: int, eps: float = 1e-6) -> nn.Module:
    """Create normalization layer by name.

    Args:
        norm_type: "none", "layer_norm", or "rms_norm"
        dim: Dimension for normalization
        eps: Epsilon value
    """
    if norm_type == "none":
        return nn.Identity()
    elif norm_type == "layer_norm":
        return nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
    elif norm_type == "rms_norm":
        return RMSNorm(dim, eps=eps)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


def create_qk_norm_layers(qk_norm: str, head_dim: int, eps: float = 1e-6):
    """Create Q/K normalization layer pair.

    Args:
        qk_norm: "none", "layer_norm", or "rms_norm"
        head_dim: Per-head dimension
        eps: Epsilon value
    Returns:
        (q_norm, k_norm) tuple
    """
    if qk_norm == "none":
        return nn.Identity(), nn.Identity()
    elif qk_norm == "layer_norm":
        return (
            nn.LayerNorm(head_dim, elementwise_affine=False, eps=eps),
            nn.LayerNorm(head_dim, elementwise_affine=False, eps=eps),
        )
    elif qk_norm == "rms_norm":
        return RMSNorm(head_dim, eps=eps), RMSNorm(head_dim, eps=eps)
    else:
        raise ValueError(f"Unknown qk_norm: {qk_norm}")
