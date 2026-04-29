"""Register ds::fused_attention_3way custom op.

Triton kernel for RMSNorm + RoPE (3-way: VL|SA|P), then F.sdpa for joint attention.
Extension of ds::fused_attention_2way for ExpandedDoubleStreamBlock.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.library.custom_op("ds::fused_attention_3way", mutates_args=())
def fused_attention_3way(
    sa_qkv: torch.Tensor,
    vl_qkv: torch.Tensor,
    p_qkv: torch.Tensor,
    q_norm_sa_weight: torch.Tensor,
    k_norm_sa_weight: torch.Tensor,
    q_norm_vl_weight: torch.Tensor,
    k_norm_vl_weight: torch.Tensor,
    q_norm_p_weight: torch.Tensor,
    k_norm_p_weight: torch.Tensor,
    sa_rope_cos: torch.Tensor,
    sa_rope_sin: torch.Tensor,
    p_rope_cos: torch.Tensor,
    p_rope_sin: torch.Tensor,
    n_sa: int,
    n_vl: int,
    n_p: int,
) -> torch.Tensor:
    """RMSNorm + RoPE (Triton 3-way) + F.sdpa for ExpandedDoubleStreamBlock.

    Args:
        sa_qkv: (n_sa, 3*inner_dim) bf16
        vl_qkv: (n_vl, 3*inner_dim) bf16
        p_qkv:  (n_p, 3*inner_dim) bf16
        q/k_norm_{sa,vl,p}_weight: (D,) bf16 — per-stream QK RMSNorm weights
        sa_rope_cos/sin: (n_sa, D//2) fp32 — SA RoPE tables
        p_rope_cos/sin:  (n_p, D//2) fp32 — P RoPE tables
        n_sa, n_vl, n_p: token counts

    Returns:
        attn: (1, n_vl + n_sa + n_p, inner_dim) bf16 in [VL | SA | P] order
    """
    from double_stream.engine.kernels.rmsnorm_rope_ds_3way import rmsnorm_rope_kernel_3way

    TOTAL = n_vl + n_sa + n_p
    N, H, D = 1536, 24, 64

    q_out = torch.empty((H, TOTAL, D), device=sa_qkv.device, dtype=torch.bfloat16)
    k_out = torch.empty((H, TOTAL, D), device=sa_qkv.device, dtype=torch.bfloat16)
    v_out = torch.empty((H, TOTAL, D), device=sa_qkv.device, dtype=torch.bfloat16)

    sa_rope_cos = sa_rope_cos.contiguous()
    sa_rope_sin = sa_rope_sin.contiguous()
    p_rope_cos = p_rope_cos.contiguous()
    p_rope_sin = p_rope_sin.contiguous()

    rmsnorm_rope_kernel_3way[lambda meta: ((TOTAL + meta["BLOCK_S"] - 1) // meta["BLOCK_S"], H)](
        q_out,
        q_out.stride(0),
        q_out.stride(1),
        q_out.stride(2),
        k_out,
        k_out.stride(0),
        k_out.stride(1),
        k_out.stride(2),
        v_out,
        v_out.stride(0),
        v_out.stride(1),
        v_out.stride(2),
        sa_qkv,
        sa_qkv.stride(0),
        sa_qkv.stride(1),
        vl_qkv,
        vl_qkv.stride(0),
        vl_qkv.stride(1),
        p_qkv,
        p_qkv.stride(0),
        p_qkv.stride(1),
        q_norm_sa_weight,
        k_norm_sa_weight,
        q_norm_vl_weight,
        k_norm_vl_weight,
        q_norm_p_weight,
        k_norm_p_weight,
        sa_rope_cos,
        sa_rope_cos.stride(0),
        sa_rope_cos.stride(1),
        sa_rope_sin,
        sa_rope_sin.stride(0),
        sa_rope_sin.stride(1),
        p_rope_cos,
        p_rope_cos.stride(0),
        p_rope_cos.stride(1),
        p_rope_sin,
        p_rope_sin.stride(0),
        p_rope_sin.stride(1),
        BLOCK_S=128,
        BLOCK_N=D,
        D=D,
        H=H,
        M=TOTAL,
        N=N,
        N_VL=n_vl,
        N_SA=n_sa,
        N_P=n_p,
    )

    attn_out = F.scaled_dot_product_attention(
        q_out.unsqueeze(0),
        k_out.unsqueeze(0),
        v_out.unsqueeze(0),
    )
    return attn_out.squeeze(0).permute(1, 0, 2).reshape(1, TOTAL, N)


@fused_attention_3way.register_fake
def _(
    sa_qkv,
    vl_qkv,
    p_qkv,
    q_norm_sa_weight,
    k_norm_sa_weight,
    q_norm_vl_weight,
    k_norm_vl_weight,
    q_norm_p_weight,
    k_norm_p_weight,
    sa_rope_cos,
    sa_rope_sin,
    p_rope_cos,
    p_rope_sin,
    n_sa,
    n_vl,
    n_p,
):
    return sa_qkv.new_empty((1, n_sa + n_vl + n_p, 1536))
