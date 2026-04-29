"""Custom-op chain for DoubleStreamBlock (2-way: VL|SA).

ds::fused_attention_2way for RMSNorm + RoPE + Attention,
ds::vl_epilogue_ln for the VL residual + LayerNorm fusion.

v1.4-v only. For v1.4-all (ExpandedDoubleStreamBlock), see
custom_expanded_double_stream_chain.py.
"""

from __future__ import annotations

from double_stream.engine.kernels.vl_epilogue_ln import fused_vl_epilogue_ln
import double_stream.engine.ops  # noqa: F401 — registers ds:: ops
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomOpDoubleStreamBlock(nn.Module):
    """Wraps DoubleStreamBlock; ds::fused_attention_2way + ds::vl_epilogue_ln."""

    def __init__(self, block, n_sa, n_vl):
        super().__init__()
        self.sa_norm1 = block.sa_norm1
        self.sa_qkv = block.sa_qkv
        self.vl_norm1 = block.vl_norm1
        self.vl_qkv = block.vl_qkv

        self.register_buffer("q_norm_sa_weight", block.q_norm_sa.weight.data)
        self.register_buffer("k_norm_sa_weight", block.k_norm_sa.weight.data)
        self.register_buffer("q_norm_vl_weight", block.q_norm_vl.weight.data)
        self.register_buffer("k_norm_vl_weight", block.k_norm_vl.weight.data)

        self.sa_proj = block.sa_proj
        self.vl_proj = block.vl_proj
        self.sa_norm2_mlp = block.sa_norm2_mlp
        self.sa_mlp = block.sa_mlp
        self.vl_norm2_mlp = block.vl_norm2_mlp
        self.vl_mlp_w12 = block.vl_mlp.w12

        self.register_buffer("vl_w3_weight", block.vl_mlp.w3.weight.data)
        self.register_buffer("vl_w3_bias", block.vl_mlp.w3.bias.data)
        VL_DIM = block.vl_proj.bias.shape[0]
        self.register_buffer(
            "_zero_vl_bias",
            torch.zeros(VL_DIM, dtype=block.vl_proj.bias.dtype, device=block.vl_proj.bias.device),
        )
        self.register_buffer(
            "_new_vl",
            torch.empty((1, n_vl, VL_DIM), dtype=torch.bfloat16, device=block.vl_proj.bias.device),
        )
        self.register_buffer(
            "_ln_vl",
            torch.empty((1, n_vl, VL_DIM), dtype=torch.bfloat16, device=block.vl_proj.bias.device),
        )

    def forward(self, sa_tokens, vl_tokens, rope_cos, rope_sin, n_sa, n_vl, precomputed_vl_ln=None):
        sa_qkv = self.sa_qkv(self.sa_norm1(sa_tokens))
        vl_norm = precomputed_vl_ln if precomputed_vl_ln is not None else self.vl_norm1(vl_tokens)
        vl_qkv = self.vl_qkv(vl_norm)

        attn = torch.ops.ds.fused_attention_2way(
            sa_qkv.squeeze(0),
            vl_qkv.squeeze(0),
            self.q_norm_sa_weight,
            self.k_norm_sa_weight,
            self.q_norm_vl_weight,
            self.k_norm_vl_weight,
            rope_cos,
            rope_sin,
            n_sa,
            n_vl,
        )
        vl_attn = attn[:, :n_vl, :]
        sa_attn = attn[:, n_vl:, :]

        sa_tokens = sa_tokens + self.sa_proj(sa_attn)
        sa_tokens = sa_tokens + self.sa_mlp(self.sa_norm2_mlp(sa_tokens))

        vl_after_attn = vl_tokens + self.vl_proj(vl_attn)
        vl_mlp_input = self.vl_norm2_mlp(vl_after_attn)
        vl_w12 = self.vl_mlp_w12(vl_mlp_input)
        x1, x2 = vl_w12.chunk(2, dim=-1)
        vl_w3_out = F.linear(F.silu(x1) * x2, self.vl_w3_weight)
        new_vl, ln_vl = fused_vl_epilogue_ln(
            vl_after_attn,
            self._zero_vl_bias,
            self.vl_w3_bias,
            vl_w3_out.squeeze(0),
            n_vl,
            vl_after_attn.shape[-1],
            new_vl_out=self._new_vl,
            ln_vl_out=self._ln_vl,
        )

        return sa_tokens, new_vl, ln_vl


class FullCustomOpDSChain(nn.Module):
    """Chain of CustomOpDoubleStreamBlock (2-way) with cross-layer VL LN."""

    def __init__(self, blocks, sa_rope_cos, sa_rope_sin, n_sa, n_vl):
        super().__init__()
        self.custom_blocks = nn.ModuleList(
            [CustomOpDoubleStreamBlock(b, n_sa=n_sa, n_vl=n_vl) for b in blocks]
        )
        self.register_buffer("sa_rope_cos", sa_rope_cos)
        self.register_buffer("sa_rope_sin", sa_rope_sin)
        self.n_sa = n_sa
        self.n_vl = n_vl

    def forward(self, sa_tokens, vl_tokens):
        ln_vl = None
        for blk in self.custom_blocks:
            sa_tokens, vl_tokens, ln_vl = blk(
                sa_tokens,
                vl_tokens,
                self.sa_rope_cos,
                self.sa_rope_sin,
                self.n_sa,
                self.n_vl,
                precomputed_vl_ln=ln_vl,
            )
        return sa_tokens, vl_tokens
