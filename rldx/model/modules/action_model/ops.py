"""MSAT utility ops: RoPE, TimestepEncoder, SwiGLUFFN, head utils (extracted from msat.py)."""

from typing import Callable, Optional

from diffusers.models.embeddings import TimestepEmbedding, Timesteps
import torch
from torch import nn
import torch.nn.functional as F


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute RoPE frequencies in complex form (Llama style).
    Args:
        dim: Head dimension (must be even)
        end: Maximum sequence length
        theta: Base frequency parameter
    Returns:
        Complex frequencies, shape (end, dim//2) as complex64
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape freqs_cis for broadcasting with x (Llama style, adapted for our use case).
    Args:
        freqs_cis: Complex frequencies, shape (B, N, D//2) as complex64
        x: Tensor to broadcast with, shape (B, H, N, D//2) as complex
    Returns:
        Reshaped freqs_cis for broadcasting, shape (B, 1, N, D//2)
    """
    ndim = x.ndim
    assert ndim == 4, f"x should have 4 dims (B, H, N, D//2), got {ndim}"
    assert freqs_cis.ndim == 3, f"freqs_cis should have 3 dims (B, N, D//2), got {freqs_cis.ndim}"
    assert freqs_cis.shape == (x.shape[0], x.shape[2], x.shape[-1]), (
        f"freqs_cis shape {freqs_cis.shape} != (B={x.shape[0]}, N={x.shape[2]}, D//2={x.shape[-1]})"
    )

    # Reshape to (B, 1, N, D//2) for broadcasting with (B, H, N, D//2)
    return freqs_cis.unsqueeze(1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE rotation to Q and K tensors using complex multiplication (Llama style).
    Args:
        xq: Query tensor, shape (B, H, N, D) where D=head_dim
        xk: Key tensor, shape (B, H, N, D) where D=head_dim
        freqs_cis: Complex frequencies, shape (N, D//2) or (1, N, D//2) as complex64
    Returns:
        Rotated Q and K tensors
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RoPEEmbedder1D(nn.Module):
    """
    Generate RoPE embeddings for 1D sequences with multiple axes (Llama style).
    For joint_attn_v2:
    - rope_sa_only: Axis 0 (dim=16) = 0 (unused), Axis 1 (dim=48) = SA sequence position (includes time tokens and action tokens)
      * Time tokens: axis 1 = 0..num_temb_tokens-1
      * Action tokens: axis 1 = num_temb_tokens..
    - rope_vl_sa: Axis 0 (dim=48) = VL sequence (Fixed), Axis 1 (dim=16) = SA sequence (Scaled)
    """

    def __init__(
        self,
        head_dim: int,
        axes_dim: list[int],
        theta: float = 10000.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        assert sum(axes_dim) == head_dim, (
            f"sum(axes_dim)={sum(axes_dim)} must equal head_dim={head_dim}"
        )
        self.head_dim = head_dim
        self.axes_dim = axes_dim
        self.theta = theta
        self.n_axes = len(axes_dim)
        self.max_seq_len = max_seq_len

        for i, axis_dim in enumerate(axes_dim):
            freqs_cis = precompute_freqs_cis(axis_dim, max_seq_len, theta)
            # persistent=False: safetensors cannot serialize complex dtypes
            # (C64/C128) and these buffers are a deterministic function of
            # (axis_dim, max_seq_len, theta) — rebuilt on every module init.
            self.register_buffer(f"freqs_cis_{i}", freqs_cis, persistent=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ids: Position IDs (B, N, n_axes)
        """
        n_axes = ids.shape[-1]
        assert n_axes == self.n_axes, f"ids last dim {n_axes} must equal n_axes {self.n_axes}"

        freqs_list = []
        for i in range(n_axes):
            freqs_cis = getattr(self, f"freqs_cis_{i}")
            pos_ids = ids[..., i]
            freqs = freqs_cis[pos_ids]
            freqs_list.append(freqs)

        return torch.cat(freqs_list, dim=-1)


# Encoders ================================================
class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim, compute_dtype=torch.float32):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps):
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)
        return timesteps_emb


# Normalization and FFN ================================================
class SwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward block from LightningDiT.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Optional[Callable[..., nn.Module]] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


# RMSNorm, create_norm_layer, create_qk_norm_layers → imported from common.py


def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    # (B, N, D) -> (B, H, N, Dh)
    B, N, D = x.shape
    Dh = D // num_heads
    return x.view(B, N, num_heads, Dh).permute(0, 2, 1, 3).contiguous()


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    # (B, H, N, Dh) -> (B, N, D)
    B, H, N, Dh = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(B, N, H * Dh)
