"""Custom-op chain for SingleStreamBlock (2-way: VL+SA).

ss::fused_attention_2way + ss::fused_epilogue_ln.

v1.4-v only. For v1.4-all (ExpandedSingleStreamBlock), see
custom_expanded_single_stream_chain.py.
"""

from __future__ import annotations

from single_stream.engine.kernels.ss_epilogue_ln import fused_ss_epilogue_ln
import single_stream.engine.ops  # noqa: F401 — registers ss:: ops
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomOpSingleStreamBlock(nn.Module):
    """Wraps SingleStreamBlock; ss::fused_attention_2way + epilogue."""

    def __init__(self, block, n_tokens):
        super().__init__()
        self.pre_norm = block.pre_norm
        self.linear1 = block.linear1
        self.register_buffer("q_norm_weight", block.q_norm.weight.data)
        self.register_buffer("k_norm_weight", block.k_norm.weight.data)
        self.mlp_proj = block.mlp_proj
        self.linear2 = block.linear2
        self.inner_dim = block.inner_dim
        device = block.linear2.weight.device
        self.register_buffer(
            "_new_hidden",
            torch.empty((1, n_tokens, self.inner_dim), dtype=torch.bfloat16, device=device),
        )
        self.register_buffer(
            "_ln_hidden",
            torch.empty((1, n_tokens, self.inner_dim), dtype=torch.bfloat16, device=device),
        )

    def forward(self, x, rope_cos, rope_sin, n_sa, precomputed_ln=None):
        x_norm = precomputed_ln if precomputed_ln is not None else self.pre_norm(x)
        qkv_mlp = self.linear1(x_norm)
        M = qkv_mlp.shape[1]

        attn_out = torch.ops.ss.fused_attention_2way(
            qkv_mlp.view(M, -1),
            self.q_norm_weight,
            self.k_norm_weight,
            rope_cos,
            rope_sin,
            n_sa,
        )

        mlp_x1, mlp_x2 = qkv_mlp[:, :, 4608:].chunk(2, dim=-1)
        mlp_out = self.mlp_proj(F.silu(mlp_x1) * mlp_x2)

        output = self.linear2(torch.cat([attn_out, mlp_out], dim=-1))
        return output  # no residual (chain handles it)


class FullCustomOpSSChain(nn.Module):
    """Chain of CustomOpSingleStreamBlock (2-way) with cross-layer epilogue LN."""

    def __init__(self, blocks, sa_rope_cos, sa_rope_sin, n_tokens, n_sa):
        super().__init__()
        self.custom_blocks = nn.ModuleList(
            [CustomOpSingleStreamBlock(b, n_tokens=n_tokens) for b in blocks]
        )
        self.register_buffer("sa_rope_cos", sa_rope_cos)
        self.register_buffer("sa_rope_sin", sa_rope_sin)
        self.n_tokens = n_tokens
        self.n_sa = n_sa

    def forward(self, x):
        precomputed_ln = None
        for i, blk in enumerate(self.custom_blocks):
            out = blk(x, self.sa_rope_cos, self.sa_rope_sin, self.n_sa, precomputed_ln)
            if i < len(self.custom_blocks) - 1:
                x, precomputed_ln = fused_ss_epilogue_ln(
                    out,
                    M=x.shape[1],
                    DIM=x.shape[-1],
                    residual=x,
                    new_hidden_out=blk._new_hidden,
                    ln_out_out=blk._ln_hidden,
                )
            else:
                x = out + x
        return x
