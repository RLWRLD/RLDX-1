"""CustomOp-based LLM decoder chain for the backbone.

CustomLLMChain:
  Chains N decoder layers + final RMSNorm into a single compilable nn.Module.
  Uses vlm::fused_attention (opaque custom op) for attention.
  All other ops (RMSNorm, MLP, residual) are standard PyTorch — torch.compile
  optimizes and fuses them automatically.

RoPE: norm weights stored per-layer, cos/signed_sin shared at chain level.
  Matches eager computation order exactly (weight × q first, then RoPE).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernels.fused_llm_attention import (
    forward as fused_llm_attention_forward,
    prepare_fused_qkv_weight,
    prepare_norm_weight_rot,
)


class TextLayerParam(nn.Module):
    """Per-layer weight container for Qwen3 decoder layer (no forward).

    Automatically unwraps VTC LayerWrapper if present.
    Stores norm weights per-layer (cos/sin are shared at chain level).
    """

    def __init__(self, layer):
        super().__init__()
        # Auto-unwrap VTC LayerWrapper → raw decoder layer
        raw = (
            layer.layer
            if hasattr(layer, "layer") and hasattr(layer, "internal_projection")
            else layer
        )
        self.pre_attn_rmsnorm = raw.input_layernorm
        self.post_attention_layernorm = raw.post_attention_layernorm
        attn = raw.self_attn
        head_dim = attn.head_dim
        self.register_buffer("qkv_weight", prepare_fused_qkv_weight(raw))
        self.register_buffer("o_proj_weight", attn.o_proj.weight.data)
        self.register_buffer("gate_proj_weight", raw.mlp.gate_proj.weight.data)
        self.register_buffer("up_proj_weight", raw.mlp.up_proj.weight.data)
        self.register_buffer("down_proj_weight", raw.mlp.down_proj.weight.data)

        # Norm weights per-layer (cos/sin shared at chain level)
        with torch.no_grad():
            self.register_buffer("q_norm_w", attn.q_norm.weight.data.clone())
            self.register_buffer(
                "q_norm_w_rot", prepare_norm_weight_rot(attn.q_norm.weight.data, head_dim)
            )
            self.register_buffer("k_norm_w", attn.k_norm.weight.data.clone())
            self.register_buffer(
                "k_norm_w_rot", prepare_norm_weight_rot(attn.k_norm.weight.data, head_dim)
            )


class CustomLLMChain(nn.Module):
    """Wraps original Qwen3 llm layer weights, uses fused_attention custom op.

    Args:
        layers: list of decoder layers (LayerWrapper auto-unwrapped by LayerParam)
        norm: final RMSNorm module
        pos_cos: (M, head_dim) bf16 — pre-computed RoPE cos (shared)
        signed_sin: (M, head_dim) bf16 — pre-computed signed sin (shared)
        apply_final_norm: if True (default), apply final RMSNorm on last layer.
            Set False for pre-compression sub-chains that return raw hidden states.
    """

    def __init__(self, layers, norm, pos_cos, signed_sin, apply_final_norm=True):
        super().__init__()
        self.layers = nn.ModuleList([TextLayerParam(layer) for layer in layers])
        self.norm = norm
        self.apply_final_norm = apply_final_norm
        self.n_layers = len(layers)
        # Shared RoPE buffers (same for all layers in this chain)
        self.register_buffer("cos", pos_cos)
        self.register_buffer("signed_sin", signed_sin)
        seq_len = pos_cos.shape[0]
        q_dim = self.layers[0].o_proj_weight.shape[1]
        hidden_dim = self.layers[0].qkv_weight.shape[1]
        self.register_buffer(
            "_deepstack_add",
            torch.zeros((1, seq_len, hidden_dim), dtype=pos_cos.dtype, device=pos_cos.device),
        )
        for layer in self.layers:
            layer.register_buffer(
                "_attn_out",
                torch.empty((seq_len, q_dim), dtype=pos_cos.dtype, device=pos_cos.device),
            )

    def forward(self, hidden_states, deepstack_features=None, deepstack_flat_indices=None):
        cos = self.cos
        ssin = self.signed_sin
        n_layers = self.n_layers

        # First layer's input LayerNorm (no previous epilogue to fuse with)
        normed = self.layers[0].pre_attn_rmsnorm(hidden_states)

        for i, layer in enumerate(self.layers):
            # 1. (normed is pre-computed from previous epilogue or first-layer norm)

            # 2. QKV projection → single fused GEMM
            B, M, D = normed.shape
            qkv = F.linear(normed.view(M, D), layer.qkv_weight)  # (M, QKV_DIM)

            # 3. Fused attention → custom op (QK RMSNorm + RoPE + Causal Attention)
            attn_out = fused_llm_attention_forward(
                qkv,
                layer.q_norm_w,
                layer.q_norm_w_rot,
                layer.k_norm_w,
                layer.k_norm_w_rot,
                cos,
                ssin,
                out=layer._attn_out,
            )  # (M, Q_DIM)

            # 4. O projection
            attn_out = F.linear(attn_out, layer.o_proj_weight).view(B, M, -1)

            # 5. Post-attention epilogue: residual add + post_attention_layernorm
            hidden_states, post_attn_normed = torch.ops.rldx_backbone.fused_epilogue_add2_rmsnorm(
                attn_out, hidden_states, layer.post_attention_layernorm.weight
            )

            # 6. MLP (SwiGLU)
            gate = F.linear(post_attn_normed, layer.gate_proj_weight)
            up = F.linear(post_attn_normed, layer.up_proj_weight)
            mlp_out = F.linear(F.silu(gate) * up, layer.down_proj_weight)

            # 7. Post-MLP epilogue: residual [+ deepstack] + next layer's input LayerNorm
            has_ds = deepstack_features is not None and i < len(deepstack_features)
            ds_add = None
            if has_ds:
                ds_add = self._deepstack_add
                ds_add.view(-1).index_copy_(
                    0,
                    deepstack_flat_indices,
                    deepstack_features[i].to(dtype=hidden_states.dtype).reshape(-1),
                )
            if i < n_layers - 1:
                # Cross-layer fusion: #7 + next #1
                next_norm_weight = self.layers[i + 1].pre_attn_rmsnorm.weight
                if has_ds:
                    hidden_states, normed = torch.ops.rldx_backbone.fused_epilogue_add3_rmsnorm(
                        mlp_out, hidden_states, ds_add, next_norm_weight
                    )
                else:
                    hidden_states, normed = torch.ops.rldx_backbone.fused_epilogue_add2_rmsnorm(
                        mlp_out, hidden_states, next_norm_weight
                    )
            else:
                # Last layer: residual add only (no next layer to fuse with)
                if has_ds:
                    hidden_states = hidden_states + mlp_out + ds_add
                else:
                    hidden_states = hidden_states + mlp_out

        # Final norm
        if self.apply_final_norm:
            hidden_states = self.norm(hidden_states)

        return hidden_states

    def forward_with_intermediates(self, hidden_states):
        """Forward with per-layer intermediate capture for diagnosis."""
        cos = self.cos
        ssin = self.signed_sin
        intermediates = []

        for i, layer in enumerate(self.layers):
            normed = layer.pre_attn_rmsnorm(hidden_states)

            B, M, D = normed.shape
            qkv = F.linear(normed.view(M, D), layer.qkv_weight)
            attn_out = torch.ops.rldx_backbone.fused_attention(
                qkv,
                layer.q_norm_w,
                layer.q_norm_w_rot,
                layer.k_norm_w,
                layer.k_norm_w_rot,
                cos,
                ssin,
            )
            attn_out = F.linear(attn_out, layer.o_proj_weight).view(B, M, -1)

            pre_mlp_hidden = hidden_states + attn_out
            post_attn_normed = layer.post_attention_layernorm(pre_mlp_hidden)

            gate = F.linear(post_attn_normed, layer.gate_proj_weight)
            up = F.linear(post_attn_normed, layer.up_proj_weight)
            mlp_out = F.linear(F.silu(gate) * up, layer.down_proj_weight)

            hidden_states = pre_mlp_hidden + mlp_out

            intermediates.append(
                {
                    "hidden_states": hidden_states.detach().clone(),
                    "pre_mlp_hidden": pre_mlp_hidden.detach().clone(),
                    "mlp_out": mlp_out.detach().clone(),
                }
            )

        output = self.norm(hidden_states)
        return output, intermediates
