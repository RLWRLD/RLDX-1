"""Register ds::fused_attention_2way custom op.

Triton kernel for RMSNorm + RoPE, then F.sdpa for attention.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.library.custom_op("ds::fused_attention_2way", mutates_args=())
def fused_attention(
    sa_qkv: torch.Tensor,
    vl_qkv: torch.Tensor,
    q_norm_sa_weight: torch.Tensor,
    k_norm_sa_weight: torch.Tensor,
    q_norm_vl_weight: torch.Tensor,
    k_norm_vl_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    n_sa: int,
    n_vl: int,
) -> torch.Tensor:
    """RMSNorm + RoPE (Triton) + F.sdpa for DoubleStreamBlock."""
    from double_stream.engine.kernels.rmsnorm_rope_ds import rmsnorm_rope_kernel

    TOTAL = n_vl + n_sa
    N, H, D = 1536, 24, 64

    q_out = torch.empty((H, TOTAL, D), device=sa_qkv.device, dtype=torch.bfloat16)
    k_out = torch.empty((H, TOTAL, D), device=sa_qkv.device, dtype=torch.bfloat16)
    v_out = torch.empty((H, TOTAL, D), device=sa_qkv.device, dtype=torch.bfloat16)

    rope_cos = rope_cos.contiguous()
    rope_sin = rope_sin.contiguous()

    rmsnorm_rope_kernel[lambda meta: ((TOTAL + meta["BLOCK_S"] - 1) // meta["BLOCK_S"], H)](
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
        q_norm_sa_weight,
        k_norm_sa_weight,
        q_norm_vl_weight,
        k_norm_vl_weight,
        rope_cos,
        rope_cos.stride(0),
        rope_cos.stride(1),
        rope_sin,
        rope_sin.stride(0),
        rope_sin.stride(1),
        BLOCK_S=128,
        BLOCK_N=D,
        D=D,
        H=H,
        M=TOTAL,
        N=N,
        N_VL=n_vl,
        N_SA=n_sa,
    )

    attn_out = F.scaled_dot_product_attention(
        q_out.unsqueeze(0),
        k_out.unsqueeze(0),
        v_out.unsqueeze(0),
    )
    return attn_out.squeeze(0).permute(1, 0, 2).reshape(1, TOTAL, N)

    # --- Alternative: fully fused Triton kernel (RMSNorm + RoPE + Attention) ---
    # from double_stream.engine.kernels.attention_fusion_ds import fused_rmsnorm_rope_attention_ds
    # k_norm = torch.empty((H, TOTAL, D), device=sa_qkv.device, dtype=torch.float32)
    # o2 = torch.empty((TOTAL, N), device=sa_qkv.device, dtype=torch.float32)
    # v = torch.empty((H, TOTAL, D), device=sa_qkv.device, dtype=torch.float32)
    # fused_rmsnorm_rope_attention_ds[...](k_norm, o2, v, sa_qkv, vl_qkv, ...)
    # o2_bf16 = o2.to(torch.bfloat16)
    # vl_attn = o2_bf16[:n_vl, :].unsqueeze(0).clone()
    # sa_attn = o2_bf16[n_vl:, :].unsqueeze(0).clone()


@fused_attention.register_fake
def _(
    sa_qkv,
    vl_qkv,
    q_norm_sa_weight,
    k_norm_sa_weight,
    q_norm_vl_weight,
    k_norm_vl_weight,
    rope_cos,
    rope_sin,
    n_sa,
    n_vl,
):
    return sa_qkv.new_empty((1, n_sa + n_vl, 1536))
