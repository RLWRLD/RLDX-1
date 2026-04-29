"""Register ss::fused_attention_3way custom op.

Triton kernel for RMSNorm + RoPE (2-way: [VL+SA | P]), then F.sdpa for joint attention.
Extension of ss::fused_attention_2way for ExpandedSingleStreamBlock.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.library.custom_op("ss::fused_attention_3way", mutates_args=())
def fused_attention_3way(
    x_qkv: torch.Tensor,
    p_qkv: torch.Tensor,
    x_q_norm_weight: torch.Tensor,
    x_k_norm_weight: torch.Tensor,
    p_q_norm_weight: torch.Tensor,
    p_k_norm_weight: torch.Tensor,
    sa_rope_cos: torch.Tensor,
    sa_rope_sin: torch.Tensor,
    p_rope_cos: torch.Tensor,
    p_rope_sin: torch.Tensor,
    n_x: int,
    n_sa: int,
    n_p: int,
) -> torch.Tensor:
    """RMSNorm + RoPE (Triton 3-way) + F.sdpa for ExpandedSingleStreamBlock.

    Args:
        x_qkv: (N_x, QKV_DIM) bf16 — VL+SA stream (from linear1, QKV portion only)
        p_qkv: (N_p, QKV_DIM) bf16 — P stream (from p_linear1, QKV portion only)
        x_q/k_norm_weight: (D,) bf16 — VL+SA QK norm weights
        p_q/k_norm_weight: (D,) bf16 — P QK norm weights
        sa_rope_cos/sin: (n_sa, D//2) fp32 — SA RoPE tables (axis0=0)
        p_rope_cos/sin:  (n_p, D//2) fp32 — P RoPE tables (axis0=1)
        n_x: VL+SA token count
        n_sa: SA portion (last n_sa of VL+SA, for RoPE)
        n_p: physics token count

    Returns:
        attn: (1, n_x + n_p, inner_dim) bf16 in [VL+SA | P] order
    """
    from single_stream.engine.kernels.rmsnorm_rope_ss_3way import rmsnorm_rope_kernel_3way

    TOTAL = n_x + n_p
    N, H, D = 1536, 24, 64

    q_out = torch.empty((H, TOTAL, D), device=x_qkv.device, dtype=torch.bfloat16)
    k_out = torch.empty((H, TOTAL, D), device=x_qkv.device, dtype=torch.bfloat16)
    v_out = torch.empty((H, TOTAL, D), device=x_qkv.device, dtype=torch.bfloat16)

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
        x_qkv,
        x_qkv.stride(0),
        x_qkv.stride(1),
        p_qkv,
        p_qkv.stride(0),
        p_qkv.stride(1),
        x_q_norm_weight,
        x_k_norm_weight,
        p_q_norm_weight,
        p_k_norm_weight,
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
        N_X=n_x,
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
    x_qkv,
    p_qkv,
    x_q_norm_weight,
    x_k_norm_weight,
    p_q_norm_weight,
    p_k_norm_weight,
    sa_rope_cos,
    sa_rope_sin,
    p_rope_cos,
    p_rope_sin,
    n_x,
    n_sa,
    n_p,
):
    return x_qkv.new_empty((1, n_x + n_p, 1536))
