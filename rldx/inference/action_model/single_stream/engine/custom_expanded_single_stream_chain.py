"""Custom-op chain for ExpandedSingleStreamBlock (3-way: [VL+SA | P]).

ss::fused_attention_3way + ss::fused_epilogue_ln (x stream only).

v1.4-all only. For v1.4-v (SingleStreamBlock), see custom_single_stream_chain.py.
"""

from __future__ import annotations

from single_stream.engine.kernels.ss_epilogue_ln import fused_ss_epilogue_ln
import single_stream.engine.ops  # noqa: F401 — registers ss:: ops
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomOpExpandedSingleStreamBlock(nn.Module):
    """Wraps ExpandedSingleStreamBlock; ss::fused_attention_3way + epilogue.

    VL+SA: pre_norm → linear1 → split QKV|MLP → attention + SwiGLU → linear2
    P:     p_pre_norm → p_linear1 → split QKV|MLP → attention + SwiGLU → p_linear2
    """

    def __init__(self, block, n_x):
        super().__init__()
        # VL+SA stream
        self.pre_norm = block.pre_norm
        self.linear1 = block.linear1
        self.register_buffer("q_norm_weight", block.q_norm.weight.data)
        self.register_buffer("k_norm_weight", block.k_norm.weight.data)
        self.mlp_proj = block.mlp_proj
        self.linear2 = block.linear2

        # P stream
        self.p_pre_norm = block.p_pre_norm
        self.p_linear1 = block.p_linear1
        self.register_buffer("p_q_norm_weight", block.p_q_norm.weight.data)
        self.register_buffer("p_k_norm_weight", block.p_k_norm.weight.data)
        self.p_mlp_proj = block.p_mlp_proj
        self.p_linear2 = block.p_linear2
        self.p_post_norm = block.p_post_norm

        self.inner_dim = block.inner_dim
        device = block.linear2.weight.device
        self.register_buffer(
            "_new_hidden",
            torch.empty((1, n_x, self.inner_dim), dtype=torch.bfloat16, device=device),
        )
        self.register_buffer(
            "_ln_hidden",
            torch.empty((1, n_x, self.inner_dim), dtype=torch.bfloat16, device=device),
        )

    def forward(
        self,
        x,
        p_tokens,
        sa_rope_cos,
        sa_rope_sin,
        p_rope_cos,
        p_rope_sin,
        n_sa,
        n_p,
        precomputed_x_ln=None,
    ):
        # VL+SA: norm → linear1 → split QKV|MLP
        x_norm = precomputed_x_ln if precomputed_x_ln is not None else self.pre_norm(x)
        x_linear1_out = self.linear1(x_norm)
        N_x = x_linear1_out.shape[1]
        x_qkv = x_linear1_out[:, :, : 3 * self.inner_dim]

        # P: norm → p_linear1 → split QKV|MLP
        p_norm = self.p_pre_norm(p_tokens)
        p_linear1_out = self.p_linear1(p_norm)
        p_qkv = p_linear1_out[:, :, : 3 * self.inner_dim]

        # ss::fused_attention_3way [VL+SA | P]
        attn = torch.ops.ss.fused_attention_3way(
            x_qkv.squeeze(0),
            p_qkv.squeeze(0),
            self.q_norm_weight,
            self.k_norm_weight,
            self.p_q_norm_weight,
            self.p_k_norm_weight,
            sa_rope_cos,
            sa_rope_sin,
            p_rope_cos,
            p_rope_sin,
            N_x,
            n_sa,
            n_p,
        )
        x_attn = attn[:, :N_x, :]
        p_attn = attn[:, N_x:, :]

        # VL+SA MLP: SwiGLU
        x_mlp_raw = x_linear1_out[:, :, 3 * self.inner_dim :]
        mlp_x1, mlp_x2 = x_mlp_raw.chunk(2, dim=-1)
        x_mlp_out = self.mlp_proj(F.silu(mlp_x1) * mlp_x2)
        x_out = self.linear2(torch.cat([x_attn, x_mlp_out], dim=-1))

        # P MLP: SwiGLU
        p_mlp_raw = p_linear1_out[:, :, 3 * self.inner_dim :]
        p_mlp_x1, p_mlp_x2 = p_mlp_raw.chunk(2, dim=-1)
        p_mlp_out = self.p_mlp_proj(F.silu(p_mlp_x1) * p_mlp_x2)
        p_out = self.p_linear2(torch.cat([p_attn, p_mlp_out], dim=-1))
        p_out = self.p_post_norm(p_out)

        return x_out, p_out  # no residual (chain handles it)


class FullExpandedCustomOpSSChain(nn.Module):
    """Chain of CustomOpExpandedSingleStreamBlock (3-way) with cross-layer epilogue LN.

    Args:
        blocks: list of ExpandedSingleStreamBlock
        sa_rope_cos/sin: (n_sa, D//2) fp32 — SA RoPE (axis0=0)
        p_rope_cos/sin: (n_p, D//2) fp32 — P RoPE (axis0=1)
        n_sa, n_p: token counts
    """

    def __init__(self, blocks, sa_rope_cos, sa_rope_sin, p_rope_cos, p_rope_sin, n_x, n_sa, n_p):
        super().__init__()
        self.custom_blocks = nn.ModuleList(
            [CustomOpExpandedSingleStreamBlock(b, n_x=n_x) for b in blocks]
        )
        self.register_buffer("sa_rope_cos", sa_rope_cos)
        self.register_buffer("sa_rope_sin", sa_rope_sin)
        self.register_buffer("p_rope_cos", p_rope_cos)
        self.register_buffer("p_rope_sin", p_rope_sin)
        self.n_x = n_x
        self.n_sa = n_sa
        self.n_p = n_p

    def forward(self, x, p_tokens):
        precomputed_x_ln = None
        for i, blk in enumerate(self.custom_blocks):
            x_out, p_out = blk(
                x,
                p_tokens,
                self.sa_rope_cos,
                self.sa_rope_sin,
                self.p_rope_cos,
                self.p_rope_sin,
                self.n_sa,
                self.n_p,
                precomputed_x_ln=precomputed_x_ln,
            )
            if i < len(self.custom_blocks) - 1:
                x, precomputed_x_ln = fused_ss_epilogue_ln(
                    x_out,
                    M=x.shape[1],
                    DIM=x.shape[-1],
                    residual=x,
                    new_hidden_out=blk._new_hidden,
                    ln_out_out=blk._ln_hidden,
                )
                p_tokens = p_tokens + p_out
            else:
                x = x_out + x
                p_tokens = p_tokens + p_out
        return x, p_tokens
