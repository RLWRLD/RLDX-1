"""MSAT stream blocks: Single/Double/Expanded/Triple (extracted from msat.py)."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from rldx.model.modules.action_model.ops import (
    SwiGLUFFN,
    _merge_heads,
    _split_heads,
    apply_rotary_emb,
)
from rldx.model.modules.norms import create_norm_layer, create_qk_norm_layers


# Deprecated shims — the action-head refactor removed the non-SwiGLU MLP
# path (`use_swiglu=False`) and the sinusoidal positional-embedding path
# (`positional_embeddings="sinusoidal"`). A handful of `if ... else:`
# branches throughout this module still reference those classes as a
# fallback, so we keep named stubs that raise at construction time to
# preserve a clean, explicit error if someone ever configures one of the
# removed options. No production config hits these paths.
class FeedForward:  # noqa: N801 — match the removed class name
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "use_swiglu=False (FeedForward path) was removed in the "
            "action-head refactor; only SwiGLU MLPs are supported."
        )


class SinusoidalPositionalEmbedding:  # noqa: N801 — match the removed class name
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "positional_embeddings='sinusoidal' was removed in the "
            "action-head refactor; use 'rope_sa_only', 'rope_vl_sa', or None."
        )


@dataclass
class ModulationOut:
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor


class Modulation(nn.Module):
    """
    Flux-style modulation for AdaLN.
    - double=True: generates 6 parameters (shift1, scale1, gate1, shift2, scale2, gate2)
    - double=False: generates 3 parameters (shift, scale, gate)
    """

    def __init__(self, dim: int, double: bool, remove_bias: bool = False):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=not remove_bias)

    def forward(self, vec: torch.Tensor) -> tuple[ModulationOut, Optional[ModulationOut]]:
        out = self.lin(F.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


# Models ================================================
class SingleStreamBlock(nn.Module):
    """
    Flux-style SingleStreamBlock with parallel linear layers.
    Processes concatenated VL+SA stream as a single stream.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        norm_eps: float = 1e-6,
        qk_norm: str = "none",
        use_swiglu: bool = False,
        positional_embeddings: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        temb_type: str = "layerwise_mod",
        remove_bias: bool = False,
        pre_norm: str = "layer_norm",
        post_norm: str = "none",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.mlp_ratio = mlp_ratio
        self.use_swiglu = use_swiglu
        self.temb_type = temb_type

        if use_swiglu:
            # For SwiGLU: mlp_hidden_dim is 2/3 of expanded dim, so we need 2 * expanded_dim for input
            mlp_expanded_dim = int(hidden_size * mlp_ratio)
            self.mlp_hidden_dim = int(
                2 / 3 * mlp_expanded_dim
            )  # Output dimension after SwiGLU (w3 output)
            mlp_in_dim = (
                2 * mlp_expanded_dim
            )  # Input dimension for SwiGLU (w12 output: 2 * expanded_dim)

            # Projection from SwiGLU output (expanded_dim) to mlp_hidden_dim (2/3 * expanded_dim)
            self.mlp_proj = nn.Linear(mlp_expanded_dim, self.mlp_hidden_dim, bias=not remove_bias)
        else:
            # For GELU: standard MLP structure
            self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
            mlp_in_dim = self.mlp_hidden_dim
            self.mlp_proj = None

        # Parallel linear layers: qkv (using inner_dim = num_heads * head_dim) and mlp_in together
        self.linear1 = nn.Linear(hidden_size, self.inner_dim * 3 + mlp_in_dim, bias=attention_bias)
        # proj (inner_dim) and mlp_out together -> hidden_size
        self.linear2 = nn.Linear(
            self.inner_dim + self.mlp_hidden_dim, hidden_size, bias=attention_bias
        )

        # QK normalization
        self.q_norm, self.k_norm = create_qk_norm_layers(qk_norm, self.head_dim, norm_eps)

        # Normalization layers
        # Pre-norm: applied before parallel computation (shared for attention and MLP)
        self.pre_norm = create_norm_layer(pre_norm, hidden_size, eps=norm_eps)
        # Post-norm: applied to the final output after linear2 (not to individual attention/MLP branches)
        self.post_norm = create_norm_layer(post_norm, hidden_size, eps=norm_eps)
        # Modulation takes inner_dim (temb dimension) and outputs hidden_size-sized parameters
        # For SingleStreamBlock, we assume hidden_size == inner_dim (from DoubleStreamBlock output)
        if temb_type != "shared_mod" and temb_type != "input_token":
            self.modulation = Modulation(hidden_size, double=False, remove_bias=remove_bias)

        # MLP activation (only used when not using SwiGLU)
        if not use_swiglu:
            if activation_fn == "gelu-approximate":
                self.mlp_act = nn.GELU(approximate="tanh")
            elif activation_fn == "geglu":
                self.mlp_act = nn.GELU()
            else:
                self.mlp_act = nn.GELU(approximate="tanh")
        else:
            self.mlp_act = None

        # Positional embedding (sinusoidal or RoPE)
        self.positional_embeddings = positional_embeddings
        if positional_embeddings == "sinusoidal" and max_seq_length is not None:
            self.pos_embed = SinusoidalPositionalEmbedding(
                hidden_size, max_seq_length=max_seq_length
            )
            self.use_rope = False
        elif positional_embeddings in ("rope_sa_only", "rope_vl_sa"):
            # RoPE will be applied in forward pass, no embedding module needed here
            self.pos_embed = None
            self.use_rope = True
        else:
            self.pos_embed = None
            self.use_rope = False

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        pe: Optional[torch.Tensor] = None,
        shared_modulation: Optional[ModulationOut] = None,
        time_token: Optional[torch.Tensor] = None,
        block_idx: int = 0,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass following Flux SingleStreamBlock pattern.
        x: (B, N, hidden_size) - concatenated VL+SA tokens (may include time_token if time_token is None here)
        temb: (B, hidden_size)
        pe: RoPE complex frequencies, shape (B, N, head_dim//2) as complex64 if use_rope else None
        shared_modulation: Optional ModulationOut (shift, scale, gate) when temb_type="shared_mod"
        time_token: Optional time token indicator - if not None, use identity modulation
        """
        # If time_token is provided (not None), use identity modulation (time_token handles conditioning)
        use_identity_mod = time_token is not None
        if use_identity_mod:
            mod = ModulationOut(
                shift=torch.zeros(x.shape[0], 1, self.hidden_size, device=x.device, dtype=x.dtype),
                scale=torch.zeros(x.shape[0], 1, self.hidden_size, device=x.device, dtype=x.dtype),
                gate=torch.ones(x.shape[0], 1, self.hidden_size, device=x.device, dtype=x.dtype),
            )
        elif self.temb_type == "shared_mod":
            if shared_modulation is None:
                raise ValueError("temb_type='shared_mod' but shared_modulation is None")
            mod = shared_modulation
        else:
            if not hasattr(self, "modulation"):
                raise AttributeError(
                    f"modulation not found. temb_type={self.temb_type} may not create modulation modules."
                )
            mod, _ = self.modulation(temb)
        # Pre-norm (shared for parallel computation)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        # Sinusoidal positional embedding is applied here (after modulation)
        if self.pos_embed is not None:
            x_mod = self.pos_embed(x_mod)

        # Parallel computation: qkv and mlp_in
        if self.use_swiglu:
            mlp_in_dim = 2 * int(self.hidden_size * self.mlp_ratio)  # 2 * expanded_dim for SwiGLU
        else:
            mlp_in_dim = self.mlp_hidden_dim  # Standard MLP dimension for GELU

        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.inner_dim, mlp_in_dim], dim=-1)

        # Split QKV and apply normalization
        q, k, v = qkv.chunk(3, dim=-1)  # (B, N, inner_dim)
        q = _split_heads(q, self.num_heads)  # (B, H, N, Dh)
        k = _split_heads(k, self.num_heads)  # (B, H, N, Dh)
        v = _split_heads(v, self.num_heads)  # (B, H, N, Dh)

        # Apply QK normalization
        B, H, N, Dh = q.shape
        q = q.reshape(B * H, N, Dh)
        k = k.reshape(B * H, N, Dh)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.reshape(B, H, N, Dh)
        k = k.reshape(B, H, N, Dh)

        # Apply RoPE to Q and K if enabled (before attention computation)
        if self.use_rope and pe is not None:
            q, k = apply_rotary_emb(q, k, pe)

        # Compute attention
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # (B, H, N, Dh)
        attn = _merge_heads(attn)  # (B, N, inner_dim)

        # Apply MLP activation
        if self.use_swiglu:
            # Apply SwiGLU: mlp is (B, N, 2 * expanded_dim), split into two parts
            mlp_x1, mlp_x2 = mlp.chunk(2, dim=-1)  # (B, N, expanded_dim), (B, N, expanded_dim)
            mlp_out = F.silu(mlp_x1) * mlp_x2  # (B, N, expanded_dim)
            # Project to mlp_hidden_dim (which is 2/3 * expanded_dim) to match SwiGLUFFN pattern
            mlp_out = self.mlp_proj(mlp_out)  # (B, N, mlp_hidden_dim)
        else:
            # Apply GELU activation
            mlp_out = self.mlp_act(mlp)  # (B, N, mlp_hidden_dim)

        # Concatenate attention and MLP outputs, then apply second linear layer
        output = self.linear2(torch.cat([attn, mlp_out], dim=-1))  # (B, N, inner_dim)
        # Apply post-norm to the final output (Z-Image style: norm after combining branches)
        output = self.post_norm(output)
        return x + mod.gate * self.dropout(output)


class DoubleStreamBlock(nn.Module):
    """
    Flux-style DoubleStreamBlock.
    Processes SA and VL streams separately but with joint attention.
    Each stream has independent modulation, norm, attention, and MLP.
    """

    def __init__(
        self,
        sa_dim: int,
        vl_dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        vl_mlp_ratio: Optional[float] = None,  # If None, use mlp_ratio
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        norm_eps: float = 1e-6,
        qk_norm: str = "none",
        use_swiglu: bool = False,
        positional_embeddings: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        temb_type: str = "layerwise_mod",
        remove_bias: bool = False,
        pre_norm: str = "layer_norm",
        post_norm: str = "none",
    ):
        super().__init__()
        self.sa_dim = sa_dim
        self.vl_dim = vl_dim
        self.num_heads = num_attention_heads
        self.head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.temb_type = temb_type
        # Use separate MLP ratio for VL stream if specified, otherwise use same as SA
        vl_mlp_ratio_actual = vl_mlp_ratio if vl_mlp_ratio is not None else mlp_ratio
        mlp_hidden_dim = int(sa_dim * mlp_ratio)

        # SA stream components
        # Modulation takes inner_dim (temb dimension) and outputs sa_dim-sized parameters
        if temb_type != "shared_mod" and temb_type != "input_token":
            self.sa_mod = Modulation(self.inner_dim, double=True, remove_bias=remove_bias)
        # Project modulation output to sa_dim if needed (not needed for input_token)
        if temb_type != "input_token":
            if self.inner_dim != sa_dim:
                self.sa_mod_proj = nn.Linear(self.inner_dim, sa_dim, bias=True)
            else:
                self.sa_mod_proj = nn.Identity()
        else:
            self.sa_mod_proj = None

        # SA stream normalization layers
        self.sa_norm1 = create_norm_layer(pre_norm, sa_dim, eps=norm_eps)  # Pre-attention
        self.sa_norm2_attn = create_norm_layer(post_norm, sa_dim, eps=norm_eps)  # Post-attention
        self.sa_norm2_mlp = create_norm_layer(pre_norm, sa_dim, eps=norm_eps)  # Pre-FFN
        self.sa_norm3_mlp = create_norm_layer(post_norm, sa_dim, eps=norm_eps)  # Post-FFN

        self.sa_qkv = nn.Linear(sa_dim, self.inner_dim * 3, bias=attention_bias)
        self.sa_proj = nn.Linear(self.inner_dim, sa_dim, bias=attention_bias)
        if use_swiglu:
            mlp_hidden_dim_sa = int(sa_dim * mlp_ratio)
            self.sa_mlp = SwiGLUFFN(sa_dim, int(2 / 3 * mlp_hidden_dim_sa))
        else:
            self.sa_mlp = FeedForward(
                sa_dim,
                dropout=dropout,
                activation_fn=activation_fn,
                inner_dim=mlp_hidden_dim,
            )

        # VL stream components
        # Modulation takes inner_dim (temb dimension) and outputs vl_dim-sized parameters
        if temb_type != "shared_mod" and temb_type != "input_token":
            self.vl_mod = Modulation(self.inner_dim, double=True, remove_bias=remove_bias)
        # Project modulation output to vl_dim if needed (not needed for input_token)
        if temb_type != "input_token":
            if self.inner_dim != vl_dim:
                self.vl_mod_proj = nn.Linear(self.inner_dim, vl_dim, bias=True)
            else:
                self.vl_mod_proj = nn.Identity()
        else:
            self.vl_mod_proj = None

        # VL stream normalization layers
        self.vl_norm1 = create_norm_layer(pre_norm, vl_dim, eps=norm_eps)  # Pre-attention
        self.vl_norm2_attn = create_norm_layer(post_norm, vl_dim, eps=norm_eps)  # Post-attention
        self.vl_norm2_mlp = create_norm_layer(pre_norm, vl_dim, eps=norm_eps)  # Pre-FFN
        self.vl_norm3_mlp = create_norm_layer(post_norm, vl_dim, eps=norm_eps)  # Post-FFN

        self.vl_qkv = nn.Linear(vl_dim, self.inner_dim * 3, bias=attention_bias)
        self.vl_proj = nn.Linear(self.inner_dim, vl_dim, bias=attention_bias)
        if use_swiglu:
            mlp_hidden_dim_vl = int(vl_dim * vl_mlp_ratio_actual)
            self.vl_mlp = SwiGLUFFN(vl_dim, int(2 / 3 * mlp_hidden_dim_vl))
        else:
            self.vl_mlp = FeedForward(
                vl_dim,
                dropout=dropout,
                activation_fn=activation_fn,
                inner_dim=int(vl_dim * vl_mlp_ratio_actual),
            )

        # QK normalization (per modality)
        self.q_norm_sa, self.k_norm_sa = create_qk_norm_layers(qk_norm, self.head_dim, norm_eps)
        self.q_norm_vl, self.k_norm_vl = create_qk_norm_layers(qk_norm, self.head_dim, norm_eps)

        # Positional embedding (sinusoidal or RoPE)
        self.positional_embeddings = positional_embeddings
        if positional_embeddings == "sinusoidal" and max_seq_length is not None:
            # Create separate positional embeddings for SA and VL streams since they may have different dimensions
            self.pos_embed_sa = SinusoidalPositionalEmbedding(sa_dim, max_seq_length=max_seq_length)
            self.pos_embed_vl = SinusoidalPositionalEmbedding(vl_dim, max_seq_length=max_seq_length)
            self.use_rope = False
        elif positional_embeddings in ("rope_sa_only", "rope_vl_sa"):
            # RoPE will be applied in forward pass, no embedding module needed here
            self.pos_embed_sa = None
            self.pos_embed_vl = None
            self.use_rope = True
        else:
            self.pos_embed_sa = None
            self.pos_embed_vl = None
            self.use_rope = False

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        sa_tokens: torch.Tensor,
        vl_tokens: torch.Tensor,
        temb: torch.Tensor,
        pe: Optional[torch.Tensor] = None,
        shared_modulations: Optional[dict] = None,
        has_time_token: bool = False,
        block_idx: int = 0,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass following Flux DoubleStreamBlock pattern.
        sa_tokens: (B, N_sa, sa_dim) - N_sa includes time_token if has_time_token=True
        vl_tokens: (B, N_vl, vl_dim)
        temb: (B, inner_dim)
        pe: RoPE complex frequencies, shape (B, N_vl + N_sa, head_dim//2) as complex64 if use_rope else None
        shared_modulations: Optional dict with 'sa_mod1_raw', 'sa_mod2_raw', 'vl_mod1_raw', 'vl_mod2_raw' keys
        has_time_token: bool flag indicating if time_token is present in sa_tokens
        encoder_attention_mask: Optional [B, N_vl] mask (1=visible, 0=masked) for VL tokens
        """
        B, N_sa, _ = sa_tokens.shape
        B2, N_vl, _ = vl_tokens.shape
        assert B == B2

        use_identity_mod = has_time_token

        # Get modulation parameters for both streams
        if use_identity_mod:
            # Use identity modulation when time_token is present
            sa_mod1 = ModulationOut(
                shift=torch.zeros(
                    B, 1, self.sa_dim, device=sa_tokens.device, dtype=sa_tokens.dtype
                ),
                scale=torch.zeros(
                    B, 1, self.sa_dim, device=sa_tokens.device, dtype=sa_tokens.dtype
                ),
                gate=torch.ones(B, 1, self.sa_dim, device=sa_tokens.device, dtype=sa_tokens.dtype),
            )
            sa_mod2 = ModulationOut(
                shift=torch.zeros(
                    B, 1, self.sa_dim, device=sa_tokens.device, dtype=sa_tokens.dtype
                ),
                scale=torch.zeros(
                    B, 1, self.sa_dim, device=sa_tokens.device, dtype=sa_tokens.dtype
                ),
                gate=torch.ones(B, 1, self.sa_dim, device=sa_tokens.device, dtype=sa_tokens.dtype),
            )
            vl_mod1 = ModulationOut(
                shift=torch.zeros(
                    B, 1, self.vl_dim, device=vl_tokens.device, dtype=vl_tokens.dtype
                ),
                scale=torch.zeros(
                    B, 1, self.vl_dim, device=vl_tokens.device, dtype=vl_tokens.dtype
                ),
                gate=torch.ones(B, 1, self.vl_dim, device=vl_tokens.device, dtype=vl_tokens.dtype),
            )
            vl_mod2 = ModulationOut(
                shift=torch.zeros(
                    B, 1, self.vl_dim, device=vl_tokens.device, dtype=vl_tokens.dtype
                ),
                scale=torch.zeros(
                    B, 1, self.vl_dim, device=vl_tokens.device, dtype=vl_tokens.dtype
                ),
                gate=torch.ones(B, 1, self.vl_dim, device=vl_tokens.device, dtype=vl_tokens.dtype),
            )
        elif self.temb_type == "shared_mod" and shared_modulations is not None:
            sa_mod1_raw = shared_modulations["sa_mod1_raw"]
            sa_mod2_raw = shared_modulations["sa_mod2_raw"]
            vl_mod1_raw = shared_modulations["vl_mod1_raw"]
            vl_mod2_raw = shared_modulations["vl_mod2_raw"]

            # Project modulation parameters to match stream dimensions
            sa_mod1 = ModulationOut(
                shift=self.sa_mod_proj(sa_mod1_raw.shift),
                scale=self.sa_mod_proj(sa_mod1_raw.scale),
                gate=self.sa_mod_proj(sa_mod1_raw.gate),
            )
            sa_mod2 = ModulationOut(
                shift=self.sa_mod_proj(sa_mod2_raw.shift),
                scale=self.sa_mod_proj(sa_mod2_raw.scale),
                gate=self.sa_mod_proj(sa_mod2_raw.gate),
            )
            vl_mod1 = ModulationOut(
                shift=self.vl_mod_proj(vl_mod1_raw.shift),
                scale=self.vl_mod_proj(vl_mod1_raw.scale),
                gate=self.vl_mod_proj(vl_mod1_raw.gate),
            )
            vl_mod2 = ModulationOut(
                shift=self.vl_mod_proj(vl_mod2_raw.shift),
                scale=self.vl_mod_proj(vl_mod2_raw.scale),
                gate=self.vl_mod_proj(vl_mod2_raw.gate),
            )
        else:
            # Layerwise modulation (temb_type="layerwise_mod")
            if not hasattr(self, "sa_mod") or not hasattr(self, "vl_mod"):
                raise AttributeError(
                    f"sa_mod or vl_mod not found. temb_type={self.temb_type} may not create modulation modules."
                )
            sa_mod1_raw, sa_mod2_raw = self.sa_mod(temb)
            vl_mod1_raw, vl_mod2_raw = self.vl_mod(temb)

            # Project modulation parameters to match stream dimensions
            sa_mod1 = ModulationOut(
                shift=self.sa_mod_proj(sa_mod1_raw.shift),
                scale=self.sa_mod_proj(sa_mod1_raw.scale),
                gate=self.sa_mod_proj(sa_mod1_raw.gate),
            )
            sa_mod2 = ModulationOut(
                shift=self.sa_mod_proj(sa_mod2_raw.shift),
                scale=self.sa_mod_proj(sa_mod2_raw.scale),
                gate=self.sa_mod_proj(sa_mod2_raw.gate),
            )
            vl_mod1 = ModulationOut(
                shift=self.vl_mod_proj(vl_mod1_raw.shift),
                scale=self.vl_mod_proj(vl_mod1_raw.scale),
                gate=self.vl_mod_proj(vl_mod1_raw.gate),
            )
            vl_mod2 = ModulationOut(
                shift=self.vl_mod_proj(vl_mod2_raw.shift),
                scale=self.vl_mod_proj(vl_mod2_raw.scale),
                gate=self.vl_mod_proj(vl_mod2_raw.gate),
            )

        # Prepare SA for attention
        sa_modulated = self.sa_norm1(sa_tokens)
        sa_modulated = (1 + sa_mod1.scale) * sa_modulated + sa_mod1.shift
        # Sinusoidal positional embedding is applied here (after modulation)
        if self.pos_embed_sa is not None:
            sa_modulated = self.pos_embed_sa(sa_modulated)
        sa_qkv = self.sa_qkv(sa_modulated)  # (B, N_sa, inner_dim * 3)
        sa_q, sa_k, sa_v = sa_qkv.chunk(3, dim=-1)  # (B, N_sa, inner_dim)
        sa_q = _split_heads(sa_q, self.num_heads)  # (B, H, N_sa, Dh)
        sa_k = _split_heads(sa_k, self.num_heads)  # (B, H, N_sa, Dh)
        sa_v = _split_heads(sa_v, self.num_heads)  # (B, H, N_sa, Dh)

        # Prepare VL for attention
        vl_modulated = self.vl_norm1(vl_tokens)
        vl_modulated = (1 + vl_mod1.scale) * vl_modulated + vl_mod1.shift
        # Sinusoidal positional embedding is applied here (after modulation)
        if self.pos_embed_vl is not None:
            vl_modulated = self.pos_embed_vl(vl_modulated)
        vl_qkv = self.vl_qkv(vl_modulated)  # (B, N_vl, inner_dim * 3)
        vl_q, vl_k, vl_v = vl_qkv.chunk(3, dim=-1)  # (B, N_vl, inner_dim)
        vl_q = _split_heads(vl_q, self.num_heads)  # (B, H, N_vl, Dh)
        vl_k = _split_heads(vl_k, self.num_heads)  # (B, H, N_vl, Dh)
        vl_v = _split_heads(vl_v, self.num_heads)  # (B, H, N_vl, Dh)

        # Apply QK normalization per modality
        B, H, _, Dh = sa_q.shape
        sa_q = sa_q.reshape(B * H, N_sa, Dh)
        sa_k = sa_k.reshape(B * H, N_sa, Dh)
        vl_q = vl_q.reshape(B * H, N_vl, Dh)
        vl_k = vl_k.reshape(B * H, N_vl, Dh)

        sa_q = self.q_norm_sa(sa_q)
        sa_k = self.k_norm_sa(sa_k)
        vl_q = self.q_norm_vl(vl_q)
        vl_k = self.k_norm_vl(vl_k)

        sa_q = sa_q.reshape(B, H, N_sa, Dh)
        sa_k = sa_k.reshape(B, H, N_sa, Dh)
        vl_q = vl_q.reshape(B, H, N_vl, Dh)
        vl_k = vl_k.reshape(B, H, N_vl, Dh)

        # Joint attention: concat Q, K, V (VL first, then time_token+SA)
        # Sequence: [VL | time_token | SA]
        q = torch.cat([vl_q, sa_q], dim=2)  # (B, H, N_vl + N_sa, Dh)
        k = torch.cat([vl_k, sa_k], dim=2)  # (B, H, N_vl + N_sa, Dh)
        v = torch.cat([vl_v, sa_v], dim=2)  # (B, H, N_vl + N_sa, Dh)

        # Apply RoPE to Q and K if enabled (before attention computation)
        if self.use_rope and pe is not None:
            q, k = apply_rotary_emb(q, k, pe)

        # Build joint attention mask from encoder_attention_mask if provided
        joint_attn_mask = None
        if encoder_attention_mask is not None:
            B_mask = encoder_attention_mask.shape[0]
            sa_ones = torch.ones(
                B_mask,
                N_sa,
                device=encoder_attention_mask.device,
                dtype=encoder_attention_mask.dtype,
            )
            kv_mask = torch.cat([encoder_attention_mask, sa_ones], dim=1)  # [B, N_vl + N_sa]
            joint_attn_mask = kv_mask[:, None, None, :]  # [B, 1, 1, N_kv]
            joint_attn_mask = torch.where(
                joint_attn_mask == 0,
                torch.tensor(float("-inf"), device=q.device, dtype=q.dtype),
                torch.tensor(0.0, device=q.device, dtype=q.dtype),
            )

        # Joint attention computation
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=joint_attn_mask
        )  # (B, H, N_vl + N_sa, Dh)

        # Split attention outputs
        vl_attn, sa_attn = (
            attn_out[:, :, :N_vl],
            attn_out[:, :, N_vl:],
        )  # (B, H, N_vl, Dh), (B, H, N_sa, Dh)

        # Project and apply post-norm if enabled, then residual with gate
        sa_attn = _merge_heads(sa_attn)  # (B, N_sa, inner_dim)
        sa_attn_proj = self.sa_proj(self.dropout(sa_attn))
        sa_attn_proj = self.sa_norm2_attn(sa_attn_proj)  # Post-attention norm
        sa_tokens = sa_tokens + sa_mod1.gate * sa_attn_proj

        vl_attn = _merge_heads(vl_attn)  # (B, N_vl, inner_dim)
        vl_attn_proj = self.vl_proj(self.dropout(vl_attn))
        vl_attn_proj = self.vl_norm2_attn(vl_attn_proj)  # Post-attention norm
        vl_tokens = vl_tokens + vl_mod1.gate * vl_attn_proj

        # MLP blocks
        sa_mlp_input = (1 + sa_mod2.scale) * self.sa_norm2_mlp(sa_tokens) + sa_mod2.shift
        sa_mlp_out = self.sa_mlp(sa_mlp_input)
        sa_mlp_out = self.sa_norm3_mlp(sa_mlp_out)  # Post-FFN norm
        sa_tokens = sa_tokens + sa_mod2.gate * sa_mlp_out

        vl_mlp_input = (1 + vl_mod2.scale) * self.vl_norm2_mlp(vl_tokens) + vl_mod2.shift
        vl_mlp_out = self.vl_mlp(vl_mlp_input)
        vl_mlp_out = self.vl_norm3_mlp(vl_mlp_out)  # Post-FFN norm
        vl_tokens = vl_tokens + vl_mod2.gate * vl_mlp_out

        return sa_tokens, vl_tokens


class ExpandedDoubleStreamBlock(DoubleStreamBlock):
    """DoubleStreamBlock extended with a P (physics) stream for 3-way joint attention.

    Inherits all SA/VL parameters from DoubleStreamBlock (same attribute names,
    so pretrained weights load directly). Adds P stream parameters (p_*).

    When p_tokens is None, delegates to DoubleStreamBlock.forward (2-way attention).
    When p_tokens is given, performs 3-way joint attention [VL | SA | P].
    """

    def __init__(
        self,
        sa_dim: int,
        vl_dim: int,
        p_dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        vl_mlp_ratio: Optional[float] = None,
        p_mlp_ratio: Optional[float] = None,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        norm_eps: float = 1e-6,
        qk_norm: str = "none",
        use_swiglu: bool = False,
        positional_embeddings: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        temb_type: str = "layerwise_mod",
        remove_bias: bool = False,
        pre_norm: str = "layer_norm",
        post_norm: str = "none",
    ):
        super().__init__(
            sa_dim=sa_dim,
            vl_dim=vl_dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            mlp_ratio=mlp_ratio,
            vl_mlp_ratio=vl_mlp_ratio,
            dropout=dropout,
            activation_fn=activation_fn,
            attention_bias=attention_bias,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            use_swiglu=use_swiglu,
            positional_embeddings=positional_embeddings,
            max_seq_length=max_seq_length,
            temb_type=temb_type,
            remove_bias=remove_bias,
            pre_norm=pre_norm,
            post_norm=post_norm,
        )
        # ── P (physics) stream ────────────────────────────────────────────
        self.p_dim = p_dim
        p_mlp_ratio_actual = p_mlp_ratio if p_mlp_ratio is not None else mlp_ratio

        if temb_type != "shared_mod" and temb_type != "input_token":
            self.p_mod = Modulation(self.inner_dim, double=True, remove_bias=remove_bias)
        if temb_type != "input_token":
            self.p_mod_proj = (
                nn.Linear(self.inner_dim, p_dim, bias=True)
                if self.inner_dim != p_dim
                else nn.Identity()
            )
        else:
            self.p_mod_proj = None

        self.p_norm1 = create_norm_layer(pre_norm, p_dim, eps=norm_eps)
        self.p_norm2_attn = create_norm_layer(post_norm, p_dim, eps=norm_eps)
        self.p_norm2_mlp = create_norm_layer(pre_norm, p_dim, eps=norm_eps)
        self.p_norm3_mlp = create_norm_layer(post_norm, p_dim, eps=norm_eps)

        self.p_qkv = nn.Linear(p_dim, self.inner_dim * 3, bias=attention_bias)
        self.p_proj = nn.Linear(self.inner_dim, p_dim, bias=attention_bias)
        if use_swiglu:
            mlp_hidden_dim_p = int(p_dim * p_mlp_ratio_actual)
            self.p_mlp = SwiGLUFFN(p_dim, int(2 / 3 * mlp_hidden_dim_p))
        else:
            self.p_mlp = FeedForward(
                p_dim,
                dropout=dropout,
                activation_fn=activation_fn,
                inner_dim=int(p_dim * p_mlp_ratio_actual),
            )

        self.q_norm_p, self.k_norm_p = create_qk_norm_layers(qk_norm, self.head_dim, norm_eps)

        if positional_embeddings == "sinusoidal" and max_seq_length is not None:
            self.pos_embed_p = SinusoidalPositionalEmbedding(p_dim, max_seq_length=max_seq_length)
        else:
            self.pos_embed_p = None

    def forward(
        self,
        sa_tokens: torch.Tensor,
        vl_tokens: torch.Tensor,
        temb: torch.Tensor,
        pe: Optional[torch.Tensor] = None,
        shared_modulations: Optional[dict] = None,
        has_time_token: bool = False,
        block_idx: int = 0,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        p_tokens: Optional[torch.Tensor] = None,
        physics_attention_mask: Optional[torch.Tensor] = None,
    ):
        if p_tokens is None:
            return super().forward(
                sa_tokens,
                vl_tokens,
                temb,
                pe=pe,
                shared_modulations=shared_modulations,
                has_time_token=has_time_token,
                block_idx=block_idx,
                encoder_attention_mask=encoder_attention_mask,
            )

        # ── 3-way attention [VL | SA | P] ────────────────────────────────
        B, N_sa, _ = sa_tokens.shape
        N_vl = vl_tokens.shape[1]
        N_p = p_tokens.shape[1]
        use_identity_mod = has_time_token

        # Modulation
        if use_identity_mod:
            sa_mod1 = sa_mod2 = ModulationOut(
                shift=torch.zeros(
                    B, 1, self.sa_dim, device=sa_tokens.device, dtype=sa_tokens.dtype
                ),
                scale=torch.zeros(
                    B, 1, self.sa_dim, device=sa_tokens.device, dtype=sa_tokens.dtype
                ),
                gate=torch.ones(B, 1, self.sa_dim, device=sa_tokens.device, dtype=sa_tokens.dtype),
            )
            vl_mod1 = vl_mod2 = ModulationOut(
                shift=torch.zeros(
                    B, 1, self.vl_dim, device=vl_tokens.device, dtype=vl_tokens.dtype
                ),
                scale=torch.zeros(
                    B, 1, self.vl_dim, device=vl_tokens.device, dtype=vl_tokens.dtype
                ),
                gate=torch.ones(B, 1, self.vl_dim, device=vl_tokens.device, dtype=vl_tokens.dtype),
            )
            p_mod1 = p_mod2 = ModulationOut(
                shift=torch.zeros(B, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
                scale=torch.zeros(B, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
                gate=torch.ones(B, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
            )
        elif self.temb_type == "shared_mod" and shared_modulations is not None:

            def _proj(proj, raw):
                return ModulationOut(
                    shift=proj(raw.shift), scale=proj(raw.scale), gate=proj(raw.gate)
                )

            sa_mod1 = _proj(self.sa_mod_proj, shared_modulations["sa_mod1_raw"])
            sa_mod2 = _proj(self.sa_mod_proj, shared_modulations["sa_mod2_raw"])
            vl_mod1 = _proj(self.vl_mod_proj, shared_modulations["vl_mod1_raw"])
            vl_mod2 = _proj(self.vl_mod_proj, shared_modulations["vl_mod2_raw"])
            p_raw1 = shared_modulations.get("p_mod1_raw")
            if p_raw1 is not None:
                p_mod1 = _proj(self.p_mod_proj, p_raw1)
                p_mod2 = _proj(self.p_mod_proj, shared_modulations["p_mod2_raw"])
            else:
                p_mod1 = p_mod2 = ModulationOut(
                    shift=torch.zeros(
                        B, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype
                    ),
                    scale=torch.zeros(
                        B, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype
                    ),
                    gate=torch.ones(B, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
                )
        else:

            def _proj(proj, raw):
                return ModulationOut(
                    shift=proj(raw.shift), scale=proj(raw.scale), gate=proj(raw.gate)
                )

            sa_mod1, sa_mod2 = self.sa_mod(temb)
            sa_mod1 = _proj(self.sa_mod_proj, sa_mod1)
            sa_mod2 = _proj(self.sa_mod_proj, sa_mod2)
            vl_mod1, vl_mod2 = self.vl_mod(temb)
            vl_mod1 = _proj(self.vl_mod_proj, vl_mod1)
            vl_mod2 = _proj(self.vl_mod_proj, vl_mod2)
            p_mod1, p_mod2 = self.p_mod(temb)
            p_mod1 = _proj(self.p_mod_proj, p_mod1)
            p_mod2 = _proj(self.p_mod_proj, p_mod2)

        # QKV preparation
        sa_mod_out = (1 + sa_mod1.scale) * self.sa_norm1(sa_tokens) + sa_mod1.shift
        if self.pos_embed_sa is not None:
            sa_mod_out = self.pos_embed_sa(sa_mod_out)
        sa_q, sa_k, sa_v = [
            _split_heads(t, self.num_heads) for t in self.sa_qkv(sa_mod_out).chunk(3, dim=-1)
        ]

        vl_mod_out = (1 + vl_mod1.scale) * self.vl_norm1(vl_tokens) + vl_mod1.shift
        if self.pos_embed_vl is not None:
            vl_mod_out = self.pos_embed_vl(vl_mod_out)
        vl_q, vl_k, vl_v = [
            _split_heads(t, self.num_heads) for t in self.vl_qkv(vl_mod_out).chunk(3, dim=-1)
        ]

        p_mod_out = (1 + p_mod1.scale) * self.p_norm1(p_tokens) + p_mod1.shift
        if self.pos_embed_p is not None:
            p_mod_out = self.pos_embed_p(p_mod_out)
        p_q, p_k, p_v = [
            _split_heads(t, self.num_heads) for t in self.p_qkv(p_mod_out).chunk(3, dim=-1)
        ]

        # QK norm — direct reassignment so the norm modules stay on the
        # autograd graph. The previous `q_t.data = ...` loop swapped storage
        # but bypassed grad_fn, leaving q_norm_*/k_norm_* weights ungradable.
        B, H, _, Dh = sa_q.shape
        sa_q = self.q_norm_sa(sa_q.reshape(B * H, N_sa, Dh)).reshape(B, H, N_sa, Dh)
        sa_k = self.k_norm_sa(sa_k.reshape(B * H, N_sa, Dh)).reshape(B, H, N_sa, Dh)
        vl_q = self.q_norm_vl(vl_q.reshape(B * H, N_vl, Dh)).reshape(B, H, N_vl, Dh)
        vl_k = self.k_norm_vl(vl_k.reshape(B * H, N_vl, Dh)).reshape(B, H, N_vl, Dh)
        p_q = self.q_norm_p(p_q.reshape(B * H, N_p, Dh)).reshape(B, H, N_p, Dh)
        p_k = self.k_norm_p(p_k.reshape(B * H, N_p, Dh)).reshape(B, H, N_p, Dh)

        # Joint attention [VL | SA | P]
        q = torch.cat([vl_q, sa_q, p_q], dim=2)
        k = torch.cat([vl_k, sa_k, p_k], dim=2)
        v = torch.cat([vl_v, sa_v, p_v], dim=2)
        if self.use_rope and pe is not None:
            q, k = apply_rotary_emb(q, k, pe)

        # Build attention mask for [VL | SA | P]
        attn_mask = None
        if encoder_attention_mask is not None or physics_attention_mask is not None:
            # VL mask
            if encoder_attention_mask is not None:
                vl_mask = encoder_attention_mask  # [B, N_vl]
            else:
                vl_mask = torch.ones(B, N_vl, device=q.device, dtype=q.dtype)
            # SA mask (always visible)
            sa_mask = torch.ones(B, N_sa, device=q.device, dtype=vl_mask.dtype)
            # P mask
            if physics_attention_mask is not None:
                p_mask = physics_attention_mask[:, None].expand(-1, N_p).to(dtype=vl_mask.dtype)
            else:
                p_mask = torch.ones(B, N_p, device=q.device, dtype=vl_mask.dtype)
            kv_mask = torch.cat([vl_mask, sa_mask, p_mask], dim=1)
            attn_mask = kv_mask[:, None, None, :]
            attn_mask = torch.where(
                attn_mask == 0,
                torch.tensor(float("-inf"), device=q.device, dtype=q.dtype),
                torch.tensor(0.0, device=q.device, dtype=q.dtype),
            )

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        vl_attn = _merge_heads(attn_out[:, :, :N_vl])
        sa_attn = _merge_heads(attn_out[:, :, N_vl : N_vl + N_sa])
        p_attn = _merge_heads(attn_out[:, :, N_vl + N_sa :])

        # Residual + post-attn norm
        sa_tokens = sa_tokens + sa_mod1.gate * self.sa_norm2_attn(
            self.sa_proj(self.dropout(sa_attn))
        )
        vl_tokens = vl_tokens + vl_mod1.gate * self.vl_norm2_attn(
            self.vl_proj(self.dropout(vl_attn))
        )
        p_tokens = p_tokens + p_mod1.gate * self.p_norm2_attn(self.p_proj(self.dropout(p_attn)))

        # MLP
        sa_tokens = sa_tokens + sa_mod2.gate * self.sa_norm3_mlp(
            self.sa_mlp((1 + sa_mod2.scale) * self.sa_norm2_mlp(sa_tokens) + sa_mod2.shift)
        )
        vl_tokens = vl_tokens + vl_mod2.gate * self.vl_norm3_mlp(
            self.vl_mlp((1 + vl_mod2.scale) * self.vl_norm2_mlp(vl_tokens) + vl_mod2.shift)
        )
        p_tokens = p_tokens + p_mod2.gate * self.p_norm3_mlp(
            self.p_mlp((1 + p_mod2.scale) * self.p_norm2_mlp(p_tokens) + p_mod2.shift)
        )

        return sa_tokens, vl_tokens, p_tokens


class ExpandedSingleStreamBlock(SingleStreamBlock):
    """SingleStreamBlock extended with a parallel P (physics) stream.

    Inherits all parameters from SingleStreamBlock (same attribute names,
    so pretrained weights load directly). Adds parallel P stream parameters (p_*).

    When p_tokens is None, delegates to SingleStreamBlock.forward.
    When p_tokens is given, does joint attention [VL+SA | P] then separate outputs.
    """

    def __init__(
        self,
        hidden_size: int,
        p_dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        norm_eps: float = 1e-6,
        qk_norm: str = "none",
        use_swiglu: bool = False,
        positional_embeddings: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        temb_type: str = "layerwise_mod",
        remove_bias: bool = False,
        pre_norm: str = "layer_norm",
        post_norm: str = "none",
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            activation_fn=activation_fn,
            attention_bias=attention_bias,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            use_swiglu=use_swiglu,
            positional_embeddings=positional_embeddings,
            max_seq_length=max_seq_length,
            temb_type=temb_type,
            remove_bias=remove_bias,
            pre_norm=pre_norm,
            post_norm=post_norm,
        )
        # ── P (physics) stream — mirrors SingleStreamBlock structure ──────
        self.p_dim = p_dim
        if use_swiglu:
            p_mlp_expanded = int(p_dim * mlp_ratio)
            self.p_mlp_hidden_dim = int(2 / 3 * p_mlp_expanded)
            p_mlp_in_dim = 2 * p_mlp_expanded
            self.p_mlp_proj = nn.Linear(p_mlp_expanded, self.p_mlp_hidden_dim, bias=not remove_bias)
        else:
            self.p_mlp_hidden_dim = int(p_dim * mlp_ratio)
            p_mlp_in_dim = self.p_mlp_hidden_dim
            self.p_mlp_proj = None

        self.p_linear1 = nn.Linear(p_dim, self.inner_dim * 3 + p_mlp_in_dim, bias=attention_bias)
        self.p_linear2 = nn.Linear(
            self.inner_dim + self.p_mlp_hidden_dim, p_dim, bias=attention_bias
        )
        self.p_q_norm, self.p_k_norm = create_qk_norm_layers(qk_norm, self.head_dim, norm_eps)
        self.p_pre_norm = create_norm_layer(pre_norm, p_dim, eps=norm_eps)
        self.p_post_norm = create_norm_layer(post_norm, p_dim, eps=norm_eps)

        if not use_swiglu:
            if activation_fn == "gelu-approximate":
                self.p_mlp_act = nn.GELU(approximate="tanh")
            else:
                self.p_mlp_act = nn.GELU(approximate="tanh")
        else:
            self.p_mlp_act = None

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        pe: Optional[torch.Tensor] = None,
        shared_modulation: Optional[ModulationOut] = None,
        time_token: Optional[torch.Tensor] = None,
        block_idx: int = 0,
        attn_mask: Optional[torch.Tensor] = None,
        p_tokens: Optional[torch.Tensor] = None,
    ):
        if p_tokens is None:
            return super().forward(
                x,
                temb,
                pe=pe,
                shared_modulation=shared_modulation,
                time_token=time_token,
                block_idx=block_idx,
                attn_mask=attn_mask,
            )

        # ── Joint attention [VL+SA | P] ──────────────────────────────────
        B, N_x, _ = x.shape
        N_p = p_tokens.shape[1]

        # VL+SA modulation (same as SingleStreamBlock)
        use_identity_mod = time_token is not None
        if use_identity_mod:
            mod = ModulationOut(
                shift=torch.zeros(B, 1, self.hidden_size, device=x.device, dtype=x.dtype),
                scale=torch.zeros(B, 1, self.hidden_size, device=x.device, dtype=x.dtype),
                gate=torch.ones(B, 1, self.hidden_size, device=x.device, dtype=x.dtype),
            )
            p_mod = ModulationOut(
                shift=torch.zeros(B, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
                scale=torch.zeros(B, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
                gate=torch.ones(B, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
            )
        elif self.temb_type == "shared_mod":
            mod = (
                shared_modulation
                if shared_modulation is not None
                else ModulationOut(
                    shift=torch.zeros(B, 1, self.hidden_size, device=x.device, dtype=x.dtype),
                    scale=torch.zeros(B, 1, self.hidden_size, device=x.device, dtype=x.dtype),
                    gate=torch.ones(B, 1, self.hidden_size, device=x.device, dtype=x.dtype),
                )
            )
            p_mod = ModulationOut(
                shift=torch.zeros(B, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
                scale=torch.zeros(B, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
                gate=torch.ones(B, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
            )
        else:
            mod, _ = self.modulation(temb)
            p_mod = ModulationOut(
                shift=torch.zeros(B, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
                scale=torch.zeros(B, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
                gate=torch.ones(B, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
            )

        # VL+SA: pre-norm -> fused QKV+MLP
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        if self.pos_embed is not None:
            x_mod = self.pos_embed(x_mod)
        if self.use_swiglu:
            x_mlp_in_dim = 2 * int(self.hidden_size * self.mlp_ratio)
        else:
            x_mlp_in_dim = self.mlp_hidden_dim
        x_qkv, x_mlp = torch.split(self.linear1(x_mod), [3 * self.inner_dim, x_mlp_in_dim], dim=-1)
        x_q, x_k, x_v = [_split_heads(t, self.num_heads) for t in x_qkv.chunk(3, dim=-1)]

        # P: pre-norm -> fused QKV+MLP
        p_mod_out = (1 + p_mod.scale) * self.p_pre_norm(p_tokens) + p_mod.shift
        if self.use_swiglu:
            p_mlp_in_dim = 2 * int(self.p_dim * self.mlp_ratio)
        else:
            p_mlp_in_dim = self.p_mlp_hidden_dim
        p_qkv, p_mlp = torch.split(
            self.p_linear1(p_mod_out), [3 * self.inner_dim, p_mlp_in_dim], dim=-1
        )
        p_q, p_k, p_v = [_split_heads(t, self.num_heads) for t in p_qkv.chunk(3, dim=-1)]

        # QK norm
        B, H, _, Dh = x_q.shape
        x_q = self.q_norm(x_q.reshape(B * H, N_x, Dh)).reshape(B, H, N_x, Dh)
        x_k = self.k_norm(x_k.reshape(B * H, N_x, Dh)).reshape(B, H, N_x, Dh)
        p_q = self.p_q_norm(p_q.reshape(B * H, N_p, Dh)).reshape(B, H, N_p, Dh)
        p_k = self.p_k_norm(p_k.reshape(B * H, N_p, Dh)).reshape(B, H, N_p, Dh)

        # Joint attention [VL+SA | P]
        q = torch.cat([x_q, p_q], dim=2)
        k = torch.cat([x_k, p_k], dim=2)
        v = torch.cat([x_v, p_v], dim=2)
        if self.use_rope and pe is not None:
            q, k = apply_rotary_emb(q, k, pe)
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x_attn = _merge_heads(attn_out[:, :, :N_x])
        p_attn = _merge_heads(attn_out[:, :, N_x:])

        # VL+SA MLP
        if self.use_swiglu:
            mlp_x1, mlp_x2 = x_mlp.chunk(2, dim=-1)
            x_mlp_out = self.mlp_proj(F.silu(mlp_x1) * mlp_x2)
        else:
            x_mlp_out = self.mlp_act(x_mlp)
        x_out = self.linear2(torch.cat([x_attn, x_mlp_out], dim=-1))
        x = x + mod.gate * self.dropout(self.post_norm(x_out))

        # P MLP
        if self.use_swiglu:
            p_mlp_x1, p_mlp_x2 = p_mlp.chunk(2, dim=-1)
            p_mlp_out = self.p_mlp_proj(F.silu(p_mlp_x1) * p_mlp_x2)
        else:
            p_mlp_out = self.p_mlp_act(p_mlp)
        p_out = self.p_linear2(torch.cat([p_attn, p_mlp_out], dim=-1))
        p_tokens = p_tokens + p_mod.gate * self.dropout(self.p_post_norm(p_out))

        return x, p_tokens


class TripleStreamBlock(nn.Module):
    """
    Flux-style TripleStreamBlock.
    Processes VL, SA (state+action), and P (physics) streams separately but with joint attention.
    Each stream has independent modulation, norm, attention, and MLP.

    Joint attention sequence order: [VL | SA | P]
    """

    def __init__(
        self,
        vl_dim: int,
        sa_dim: int,
        p_dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        vl_mlp_ratio: Optional[float] = None,
        sa_mlp_ratio: Optional[float] = None,
        p_mlp_ratio: Optional[float] = None,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        norm_eps: float = 1e-6,
        qk_norm: str = "none",
        use_swiglu: bool = False,
        positional_embeddings: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        temb_type: str = "layerwise_mod",
        remove_bias: bool = False,
        pre_norm: str = "layer_norm",
        post_norm: str = "none",
    ):
        super().__init__()
        self.vl_dim = vl_dim
        self.sa_dim = sa_dim
        self.p_dim = p_dim
        self.num_heads = num_attention_heads
        self.head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.temb_type = temb_type

        # Use separate MLP ratios if specified, otherwise use mlp_ratio
        vl_mlp_ratio_actual = vl_mlp_ratio if vl_mlp_ratio is not None else mlp_ratio
        sa_mlp_ratio_actual = sa_mlp_ratio if sa_mlp_ratio is not None else mlp_ratio
        p_mlp_ratio_actual = p_mlp_ratio if p_mlp_ratio is not None else mlp_ratio

        # VL stream components
        if temb_type != "shared_mod" and temb_type != "input_token":
            self.vl_mod = Modulation(self.inner_dim, double=True, remove_bias=remove_bias)
        if self.inner_dim != vl_dim:
            self.vl_mod_proj = nn.Linear(self.inner_dim, vl_dim, bias=True)
        else:
            self.vl_mod_proj = nn.Identity()

        # VL stream normalization layers
        self.vl_norm1 = create_norm_layer(pre_norm, vl_dim, eps=norm_eps)  # Pre-attention
        self.vl_norm2_attn = create_norm_layer(post_norm, vl_dim, eps=norm_eps)  # Post-attention
        self.vl_norm2_mlp = create_norm_layer(pre_norm, vl_dim, eps=norm_eps)  # Pre-FFN
        self.vl_norm3_mlp = create_norm_layer(post_norm, vl_dim, eps=norm_eps)  # Post-FFN

        self.vl_qkv = nn.Linear(vl_dim, self.inner_dim * 3, bias=attention_bias)
        self.vl_proj = nn.Linear(self.inner_dim, vl_dim, bias=attention_bias)
        if use_swiglu:
            mlp_hidden_dim_vl = int(vl_dim * vl_mlp_ratio_actual)
            self.vl_mlp = SwiGLUFFN(vl_dim, int(2 / 3 * mlp_hidden_dim_vl))
        else:
            self.vl_mlp = FeedForward(
                vl_dim,
                dropout=dropout,
                activation_fn=activation_fn,
                inner_dim=int(vl_dim * vl_mlp_ratio_actual),
            )

        # SA (state+action) stream components
        if temb_type != "shared_mod" and temb_type != "input_token":
            self.sa_mod = Modulation(self.inner_dim, double=True, remove_bias=remove_bias)
        if temb_type != "input_token":
            if self.inner_dim != sa_dim:
                self.sa_mod_proj = nn.Linear(self.inner_dim, sa_dim, bias=True)
            else:
                self.sa_mod_proj = nn.Identity()
        else:
            self.sa_mod_proj = None

        self.sa_norm1 = create_norm_layer(pre_norm, sa_dim, eps=norm_eps)
        self.sa_norm2_attn = create_norm_layer(post_norm, sa_dim, eps=norm_eps)
        self.sa_norm2_mlp = create_norm_layer(pre_norm, sa_dim, eps=norm_eps)
        self.sa_norm3_mlp = create_norm_layer(post_norm, sa_dim, eps=norm_eps)

        self.sa_qkv = nn.Linear(sa_dim, self.inner_dim * 3, bias=attention_bias)
        self.sa_proj = nn.Linear(self.inner_dim, sa_dim, bias=attention_bias)
        if use_swiglu:
            mlp_hidden_dim_sa = int(sa_dim * sa_mlp_ratio_actual)
            self.sa_mlp = SwiGLUFFN(sa_dim, int(2 / 3 * mlp_hidden_dim_sa))
        else:
            self.sa_mlp = FeedForward(
                sa_dim,
                dropout=dropout,
                activation_fn=activation_fn,
                inner_dim=int(sa_dim * sa_mlp_ratio_actual),
            )

        # P (physics) stream components
        if temb_type != "shared_mod" and temb_type != "input_token":
            self.p_mod = Modulation(self.inner_dim, double=True, remove_bias=remove_bias)
        if temb_type != "input_token":
            if self.inner_dim != p_dim:
                self.p_mod_proj = nn.Linear(self.inner_dim, p_dim, bias=True)
            else:
                self.p_mod_proj = nn.Identity()
        else:
            self.p_mod_proj = None

        self.p_norm1 = create_norm_layer(pre_norm, p_dim, eps=norm_eps)
        self.p_norm2_attn = create_norm_layer(post_norm, p_dim, eps=norm_eps)
        self.p_norm2_mlp = create_norm_layer(pre_norm, p_dim, eps=norm_eps)
        self.p_norm3_mlp = create_norm_layer(post_norm, p_dim, eps=norm_eps)

        self.p_qkv = nn.Linear(p_dim, self.inner_dim * 3, bias=attention_bias)
        self.p_proj = nn.Linear(self.inner_dim, p_dim, bias=attention_bias)
        if use_swiglu:
            mlp_hidden_dim_p = int(p_dim * p_mlp_ratio_actual)
            self.p_mlp = SwiGLUFFN(p_dim, int(2 / 3 * mlp_hidden_dim_p))
        else:
            self.p_mlp = FeedForward(
                p_dim,
                dropout=dropout,
                activation_fn=activation_fn,
                inner_dim=int(p_dim * p_mlp_ratio_actual),
            )

        # QK normalization (per modality)
        self.q_norm_vl, self.k_norm_vl = create_qk_norm_layers(qk_norm, self.head_dim, norm_eps)
        self.q_norm_sa, self.k_norm_sa = create_qk_norm_layers(qk_norm, self.head_dim, norm_eps)
        self.q_norm_p, self.k_norm_p = create_qk_norm_layers(qk_norm, self.head_dim, norm_eps)

        # Positional embedding (RoPE or disabled)
        self.positional_embeddings = positional_embeddings
        self.pos_embed_vl = None
        self.pos_embed_sa = None
        self.pos_embed_p = None
        if positional_embeddings in ("rope_sa_only", "rope_vl_sa"):
            self.use_rope = True
        else:
            self.use_rope = False

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        vl_tokens: torch.Tensor,
        sa_tokens: torch.Tensor,
        p_tokens: torch.Tensor,
        temb: torch.Tensor,
        pe: Optional[torch.Tensor] = None,
        shared_modulations: Optional[dict] = None,
        has_time_token: bool = False,
    ):
        """
        Forward pass following Flux TripleStreamBlock pattern.
        vl_tokens: (B, N_vl, vl_dim) — vision-language features
        sa_tokens: (B, N_sa, sa_dim) — state+action features (may include time_token at the beginning)
        p_tokens:  (B, N_p, p_dim)   — physics signal features
        temb: (B, inner_dim)
        pe: RoPE complex frequencies for joint sequence [VL | SA | P]
        shared_modulations: Optional dict with 'vl_mod1_raw', 'vl_mod2_raw', 'sa_mod1_raw', 'sa_mod2_raw',
                            'p_mod1_raw', 'p_mod2_raw' keys
        has_time_token: bool flag indicating if time_token is present in sa_tokens
        """
        B, N_vl, _ = vl_tokens.shape
        B2, N_sa, _ = sa_tokens.shape
        B3, N_p, _ = p_tokens.shape
        assert B == B2 == B3

        use_identity_mod = has_time_token

        # Get modulation parameters for all streams
        if use_identity_mod:
            vl_mod1 = ModulationOut(
                shift=torch.zeros(
                    B, 1, self.vl_dim, device=vl_tokens.device, dtype=vl_tokens.dtype
                ),
                scale=torch.zeros(
                    B, 1, self.vl_dim, device=vl_tokens.device, dtype=vl_tokens.dtype
                ),
                gate=torch.ones(B, 1, self.vl_dim, device=vl_tokens.device, dtype=vl_tokens.dtype),
            )
            vl_mod2 = ModulationOut(
                shift=torch.zeros(
                    B, 1, self.vl_dim, device=vl_tokens.device, dtype=vl_tokens.dtype
                ),
                scale=torch.zeros(
                    B, 1, self.vl_dim, device=vl_tokens.device, dtype=vl_tokens.dtype
                ),
                gate=torch.ones(B, 1, self.vl_dim, device=vl_tokens.device, dtype=vl_tokens.dtype),
            )
            sa_mod1 = ModulationOut(
                shift=torch.zeros(
                    B, 1, self.sa_dim, device=sa_tokens.device, dtype=sa_tokens.dtype
                ),
                scale=torch.zeros(
                    B, 1, self.sa_dim, device=sa_tokens.device, dtype=sa_tokens.dtype
                ),
                gate=torch.ones(B, 1, self.sa_dim, device=sa_tokens.device, dtype=sa_tokens.dtype),
            )
            sa_mod2 = ModulationOut(
                shift=torch.zeros(
                    B, 1, self.sa_dim, device=sa_tokens.device, dtype=sa_tokens.dtype
                ),
                scale=torch.zeros(
                    B, 1, self.sa_dim, device=sa_tokens.device, dtype=sa_tokens.dtype
                ),
                gate=torch.ones(B, 1, self.sa_dim, device=sa_tokens.device, dtype=sa_tokens.dtype),
            )
            p_mod1 = ModulationOut(
                shift=torch.zeros(B, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
                scale=torch.zeros(B, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
                gate=torch.ones(B, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
            )
            p_mod2 = ModulationOut(
                shift=torch.zeros(B, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
                scale=torch.zeros(B, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
                gate=torch.ones(B, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
            )
        elif self.temb_type == "shared_mod" and shared_modulations is not None:
            vl_mod1_raw = shared_modulations["vl_mod1_raw"]
            vl_mod2_raw = shared_modulations["vl_mod2_raw"]
            sa_mod1_raw = shared_modulations["sa_mod1_raw"]
            sa_mod2_raw = shared_modulations["sa_mod2_raw"]
            p_mod1_raw = shared_modulations["p_mod1_raw"]
            p_mod2_raw = shared_modulations["p_mod2_raw"]

            vl_mod1 = ModulationOut(
                shift=self.vl_mod_proj(vl_mod1_raw.shift),
                scale=self.vl_mod_proj(vl_mod1_raw.scale),
                gate=self.vl_mod_proj(vl_mod1_raw.gate),
            )
            vl_mod2 = ModulationOut(
                shift=self.vl_mod_proj(vl_mod2_raw.shift),
                scale=self.vl_mod_proj(vl_mod2_raw.scale),
                gate=self.vl_mod_proj(vl_mod2_raw.gate),
            )
            sa_mod1 = ModulationOut(
                shift=self.sa_mod_proj(sa_mod1_raw.shift),
                scale=self.sa_mod_proj(sa_mod1_raw.scale),
                gate=self.sa_mod_proj(sa_mod1_raw.gate),
            )
            sa_mod2 = ModulationOut(
                shift=self.sa_mod_proj(sa_mod2_raw.shift),
                scale=self.sa_mod_proj(sa_mod2_raw.scale),
                gate=self.sa_mod_proj(sa_mod2_raw.gate),
            )
            p_mod1 = ModulationOut(
                shift=self.p_mod_proj(p_mod1_raw.shift),
                scale=self.p_mod_proj(p_mod1_raw.scale),
                gate=self.p_mod_proj(p_mod1_raw.gate),
            )
            p_mod2 = ModulationOut(
                shift=self.p_mod_proj(p_mod2_raw.shift),
                scale=self.p_mod_proj(p_mod2_raw.scale),
                gate=self.p_mod_proj(p_mod2_raw.gate),
            )
        else:
            if (
                not hasattr(self, "vl_mod")
                or not hasattr(self, "sa_mod")
                or not hasattr(self, "p_mod")
            ):
                raise AttributeError(
                    f"Modulation modules not found. temb_type={self.temb_type} may not create modulation modules."
                )
            vl_mod1_raw, vl_mod2_raw = self.vl_mod(temb)
            sa_mod1_raw, sa_mod2_raw = self.sa_mod(temb)
            p_mod1_raw, p_mod2_raw = self.p_mod(temb)

            vl_mod1 = ModulationOut(
                shift=self.vl_mod_proj(vl_mod1_raw.shift),
                scale=self.vl_mod_proj(vl_mod1_raw.scale),
                gate=self.vl_mod_proj(vl_mod1_raw.gate),
            )
            vl_mod2 = ModulationOut(
                shift=self.vl_mod_proj(vl_mod2_raw.shift),
                scale=self.vl_mod_proj(vl_mod2_raw.scale),
                gate=self.vl_mod_proj(vl_mod2_raw.gate),
            )
            sa_mod1 = ModulationOut(
                shift=self.sa_mod_proj(sa_mod1_raw.shift),
                scale=self.sa_mod_proj(sa_mod1_raw.scale),
                gate=self.sa_mod_proj(sa_mod1_raw.gate),
            )
            sa_mod2 = ModulationOut(
                shift=self.sa_mod_proj(sa_mod2_raw.shift),
                scale=self.sa_mod_proj(sa_mod2_raw.scale),
                gate=self.sa_mod_proj(sa_mod2_raw.gate),
            )
            p_mod1 = ModulationOut(
                shift=self.p_mod_proj(p_mod1_raw.shift),
                scale=self.p_mod_proj(p_mod1_raw.scale),
                gate=self.p_mod_proj(p_mod1_raw.gate),
            )
            p_mod2 = ModulationOut(
                shift=self.p_mod_proj(p_mod2_raw.shift),
                scale=self.p_mod_proj(p_mod2_raw.scale),
                gate=self.p_mod_proj(p_mod2_raw.gate),
            )

        # Prepare VL for attention
        vl_modulated = self.vl_norm1(vl_tokens)
        vl_modulated = (1 + vl_mod1.scale) * vl_modulated + vl_mod1.shift
        if self.pos_embed_vl is not None:
            vl_modulated = self.pos_embed_vl(vl_modulated)
        vl_qkv = self.vl_qkv(vl_modulated)
        vl_q, vl_k, vl_v = vl_qkv.chunk(3, dim=-1)
        vl_q = _split_heads(vl_q, self.num_heads)
        vl_k = _split_heads(vl_k, self.num_heads)
        vl_v = _split_heads(vl_v, self.num_heads)

        # Prepare SA (state+action) for attention
        # sa_tokens already includes time_token at the beginning if provided: [time_token | sa_tokens]
        sa_modulated = self.sa_norm1(sa_tokens)
        sa_modulated = (1 + sa_mod1.scale) * sa_modulated + sa_mod1.shift
        if self.pos_embed_sa is not None:
            sa_modulated = self.pos_embed_sa(sa_modulated)
        sa_qkv = self.sa_qkv(sa_modulated)
        sa_q, sa_k, sa_v = sa_qkv.chunk(3, dim=-1)
        sa_q = _split_heads(sa_q, self.num_heads)
        sa_k = _split_heads(sa_k, self.num_heads)
        sa_v = _split_heads(sa_v, self.num_heads)

        # Prepare P (physics) for attention
        p_modulated = self.p_norm1(p_tokens)
        p_modulated = (1 + p_mod1.scale) * p_modulated + p_mod1.shift
        if self.pos_embed_p is not None:
            p_modulated = self.pos_embed_p(p_modulated)
        p_qkv = self.p_qkv(p_modulated)
        p_q, p_k, p_v = p_qkv.chunk(3, dim=-1)
        p_q = _split_heads(p_q, self.num_heads)
        p_k = _split_heads(p_k, self.num_heads)
        p_v = _split_heads(p_v, self.num_heads)

        # Apply QK normalization per modality
        B, H, _, Dh = vl_q.shape
        vl_q = vl_q.reshape(B * H, N_vl, Dh)
        vl_k = vl_k.reshape(B * H, N_vl, Dh)
        sa_q = sa_q.reshape(B * H, N_sa, Dh)
        sa_k = sa_k.reshape(B * H, N_sa, Dh)
        p_q = p_q.reshape(B * H, N_p, Dh)
        p_k = p_k.reshape(B * H, N_p, Dh)

        vl_q = self.q_norm_vl(vl_q)
        vl_k = self.k_norm_vl(vl_k)
        sa_q = self.q_norm_sa(sa_q)
        sa_k = self.k_norm_sa(sa_k)
        p_q = self.q_norm_p(p_q)
        p_k = self.k_norm_p(p_k)

        vl_q = vl_q.reshape(B, H, N_vl, Dh)
        vl_k = vl_k.reshape(B, H, N_vl, Dh)
        sa_q = sa_q.reshape(B, H, N_sa, Dh)
        sa_k = sa_k.reshape(B, H, N_sa, Dh)
        p_q = p_q.reshape(B, H, N_p, Dh)
        p_k = p_k.reshape(B, H, N_p, Dh)

        # Joint attention: [VL | SA | P]
        q = torch.cat([vl_q, sa_q, p_q], dim=2)
        k = torch.cat([vl_k, sa_k, p_k], dim=2)
        v = torch.cat([vl_v, sa_v, p_v], dim=2)

        # Apply RoPE to Q and K if enabled
        if self.use_rope and pe is not None:
            q, k = apply_rotary_emb(q, k, pe)

        # Joint attention computation
        attn_out = F.scaled_dot_product_attention(q, k, v)

        # Split attention outputs: [VL | SA | P]
        vl_attn = attn_out[:, :, :N_vl]
        sa_attn = attn_out[:, :, N_vl : N_vl + N_sa]
        p_attn = attn_out[:, :, N_vl + N_sa :]

        # Project and apply post-norm, then residual with gate
        vl_attn = _merge_heads(vl_attn)
        vl_attn_proj = self.vl_proj(self.dropout(vl_attn))
        vl_attn_proj = self.vl_norm2_attn(vl_attn_proj)
        vl_tokens = vl_tokens + vl_mod1.gate * vl_attn_proj

        sa_attn = _merge_heads(sa_attn)
        sa_attn_proj = self.sa_proj(self.dropout(sa_attn))
        sa_attn_proj = self.sa_norm2_attn(sa_attn_proj)
        sa_tokens = sa_tokens + sa_mod1.gate * sa_attn_proj

        p_attn = _merge_heads(p_attn)
        p_attn_proj = self.p_proj(self.dropout(p_attn))
        p_attn_proj = self.p_norm2_attn(p_attn_proj)
        p_tokens = p_tokens + p_mod1.gate * p_attn_proj

        # MLP blocks
        vl_mlp_input = (1 + vl_mod2.scale) * self.vl_norm2_mlp(vl_tokens) + vl_mod2.shift
        vl_mlp_out = self.vl_mlp(vl_mlp_input)
        vl_mlp_out = self.vl_norm3_mlp(vl_mlp_out)
        vl_tokens = vl_tokens + vl_mod2.gate * vl_mlp_out

        sa_mlp_input = (1 + sa_mod2.scale) * self.sa_norm2_mlp(sa_tokens) + sa_mod2.shift
        sa_mlp_out = self.sa_mlp(sa_mlp_input)
        sa_mlp_out = self.sa_norm3_mlp(sa_mlp_out)
        sa_tokens = sa_tokens + sa_mod2.gate * sa_mlp_out

        p_mlp_input = (1 + p_mod2.scale) * self.p_norm2_mlp(p_tokens) + p_mod2.shift
        p_mlp_out = self.p_mlp(p_mlp_input)
        p_mlp_out = self.p_norm3_mlp(p_mlp_out)
        p_tokens = p_tokens + p_mod2.gate * p_mlp_out

        return vl_tokens, sa_tokens, p_tokens
