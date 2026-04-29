import torch
import torch.nn as nn
import torch.nn.functional as F

from rldx.model.modules.embodiment_conditioned_mlp import SinusoidalPositionalEncoding
from rldx.utils.dist import rank_zero_print as _print


class PhysicalSignalEncoder(nn.Module):
    """Encode physics history tokens: (B, T_hist, input_dim) -> (B, T_hist, output_dim)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.W2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.W3 = nn.Linear(hidden_dim, output_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.W1(x)
        t_ids = torch.arange(T, device=x.device).float().unsqueeze(0).expand(B, -1)
        pos = self.pos_encoding(t_ids).to(dtype=h.dtype)
        h = F.silu(self.W2(torch.cat([h, pos], dim=-1)))
        return self.W3(h)  # (B, T_hist, output_dim)


class PhysicalSignalDecoder(nn.Module):
    """Decode physics predictions: (B, T, input_dim) -> (B, T, output_dim)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, T, output_dim)


class PhysicsNoiseEncoder(nn.Module):
    """Future tokens: (B, T_fut, input_dim) -> (B, T_fut, output_dim)
    Positional encoding uses diffusion timestep instead of sequential index."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.W2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.W3 = nn.Linear(hidden_dim, output_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T_fut, input_dim) - noisy future physics
            timesteps: (B,) - diffusion timestep (scalar per sample)
        Returns:
            (B, T_fut, output_dim)
        """
        B, T, _ = x.shape
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)  # (B, T_fut)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,)")
        x_emb = self.W1(x)  # (B, T_fut, hidden_dim)
        t_emb = self.pos_encoding(timesteps).to(dtype=x_emb.dtype)  # (B, T_fut, hidden_dim)
        x = F.silu(self.W2(torch.cat([x_emb, t_emb], dim=-1)))  # (B, T_fut, hidden_dim)
        return self.W3(x)  # (B, T_fut, output_dim)


def _xavier(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def _small_noise(m: nn.Linear, std: float) -> None:
    nn.init.normal_(m.weight, mean=0.0, std=std)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


def _reset_norm_identity(m: nn.Module) -> None:
    """Reset LayerNorm or RMSNorm to identity (weight=1, bias=0)."""
    if isinstance(m, nn.LayerNorm):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif hasattr(m, "weight") and isinstance(m.weight, nn.Parameter):
        # Covers RMSNorm (weight only, no bias)
        nn.init.ones_(m.weight)


def _last_linear(module: nn.Module) -> nn.Linear | None:
    linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
    return linears[-1] if linears else None


def init_physics_params_near_zero(action_model: nn.Module) -> None:
    """Apply exit-zero initialization to all physics stream parameters.

    Design: internal layers get Xavier (gradient flow), exit layers get near-zero
    so that the physics stream outputs ~0 at Day-0 and does not disturb the
    pretrained action stream.
    """
    from rldx.model.modules.action_model.msat import (
        ExpandedDoubleStreamBlock,
        ExpandedSingleStreamBlock,
    )

    msat = action_model.model

    # Support both old layout (action_model.physics_cond_encoder)
    # and new layout (action_model.physics.physics_cond_encoder)
    physics_owner = getattr(action_model, "physics", None) or action_model

    # ── (A) Encoder: W1,W2 = Xavier, W3 = near-zero (exit) ──
    # Note: physics_fut_encoder also gets near-zero init (unlike reference which uses
    # PyTorch default). This is a design choice to ensure the future physics stream
    # outputs ~0 at Day-0, combined with NewParamWarmupCallback for gradual fade-in.
    for enc_name in ("physics_cond_encoder", "physics_fut_encoder"):
        if hasattr(physics_owner, enc_name):
            enc = getattr(physics_owner, enc_name)
            _xavier(enc.W1)
            _xavier(enc.W2)
            _small_noise(enc.W3, std=1e-5)
            _print(f"   [Physics init] {enc_name}: W1,W2=Xavier, W3=near-zero(1e-5)")

    # ── (A) Decoder: first Linear = Kaiming (keep), last = near-zero (exit) ──
    if hasattr(physics_owner, "physics_decoder"):
        last = _last_linear(physics_owner.physics_decoder.net)
        if last is not None:
            _small_noise(last, std=1e-4)
        _print("   [Physics init] decoder: last_linear=near-zero(1e-4)")

    # ── (B) ExpandedDoubleStreamBlock — P stream ──
    n_double = 0
    for blk in msat.double_blocks:
        if not isinstance(blk, ExpandedDoubleStreamBlock):
            continue
        n_double += 1

        # p_qkv: Xavier
        _xavier(blk.p_qkv)

        # p_proj: near-zero (exit)
        _small_noise(blk.p_proj, std=1e-4)

        # p_mlp: internal Xavier, last linear near-zero (exit)
        mlp_linears = [m for m in blk.p_mlp.modules() if isinstance(m, nn.Linear)]
        for lin in mlp_linears:
            _xavier(lin)
        if mlp_linears:
            _small_noise(mlp_linears[-1], std=1e-4)

        # p_mod: Xavier
        if hasattr(blk, "p_mod"):
            blk.p_mod.apply(_xavier)

        # Norms: identity reset (LayerNorm and RMSNorm)
        for attr in ("p_norm1", "p_norm2_attn", "p_norm2_mlp", "p_norm3_mlp"):
            m = getattr(blk, attr, None)
            if m is not None:
                _reset_norm_identity(m)

        # QK norms: identity reset (matches wip TripleStreamBlock behaviour —
        # _init_layernorm_identity is called but has no effect on RMSNorm;
        # we keep the same semantics: leave at default weight=1)
        for attr in ("q_norm_p", "k_norm_p"):
            m = getattr(blk, attr, None)
            if m is not None:
                _reset_norm_identity(m)

    if n_double > 0:
        _print(
            f"   [Physics init] {n_double} ExpandedDoubleStreamBlocks: "
            f"p_qkv=Xavier, p_proj=near-zero, p_mlp exit=near-zero"
        )

    # ── (C) ExpandedSingleStreamBlock — P stream ──
    n_single = 0
    for blk in msat.single_blocks:
        if not isinstance(blk, ExpandedSingleStreamBlock):
            continue
        n_single += 1

        # p_linear1: Xavier
        _xavier(blk.p_linear1)

        # p_linear2: near-zero (exit)
        _small_noise(blk.p_linear2, std=1e-4)

        # p_mlp_proj: Xavier (if SwiGLU)
        if getattr(blk, "p_mlp_proj", None) is not None:
            _xavier(blk.p_mlp_proj)

        # Norms: identity reset
        if hasattr(blk, "p_pre_norm"):
            _reset_norm_identity(blk.p_pre_norm)
        if hasattr(blk, "p_post_norm"):
            _reset_norm_identity(blk.p_post_norm)

        # QK norms: Xavier (matches wip DoubleStreamUpperBlock behaviour)
        for attr in ("p_q_norm", "p_k_norm"):
            m = getattr(blk, attr, None)
            if m is not None and hasattr(m, "weight"):
                _xavier(m)

    if n_single > 0:
        _print(
            f"   [Physics init] {n_single} ExpandedSingleStreamBlocks: "
            f"p_linear1=Xavier, p_linear2=near-zero"
        )

    # ── (D) MSAT physics output projection ──
    if hasattr(msat, "proj_out_physics_1"):
        _small_noise(msat.proj_out_physics_1, std=1e-5)
        _print("   [Physics init] proj_out_physics_1=near-zero(1e-5)")
    if hasattr(msat, "proj_out_physics_2"):
        _small_noise(msat.proj_out_physics_2, std=1e-4)
        _print("   [Physics init] proj_out_physics_2=near-zero(1e-4)")
