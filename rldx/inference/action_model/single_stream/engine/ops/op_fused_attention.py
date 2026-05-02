"""Register ss::fused_attention_2way custom op.

Triton kernel for RMSNorm + RoPE, then F.sdpa for attention.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.library.custom_op("ss::fused_attention_2way", mutates_args=())
def fused_attention(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    n_sa: int,
) -> torch.Tensor:
    """RMSNorm + RoPE (Triton) + F.sdpa for SingleStreamBlock."""
    from single_stream.engine.kernels.rmsnorm_rope_ss import rmsnorm_rope_kernel

    M = qkv.shape[0]
    N, H, D = 1536, 24, 64

    q_out = torch.empty((H, M, D), device=qkv.device, dtype=torch.bfloat16)
    k_out = torch.empty((H, M, D), device=qkv.device, dtype=torch.bfloat16)
    v_out = torch.empty((H, M, D), device=qkv.device, dtype=torch.bfloat16)

    rope_cos = rope_cos.contiguous()
    rope_sin = rope_sin.contiguous()

    rmsnorm_rope_kernel[lambda meta: ((M + meta["BLOCK_S"] - 1) // meta["BLOCK_S"], H)](
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
        qkv,
        qkv.stride(0),
        qkv.stride(1),
        q_norm_weight,
        k_norm_weight,
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
        M=M,
        N=N,
        N_SA=n_sa,
    )

    attn_out = F.scaled_dot_product_attention(
        q_out.unsqueeze(0),
        k_out.unsqueeze(0),
        v_out.unsqueeze(0),
    )
    return attn_out.squeeze(0).permute(1, 0, 2).contiguous().view(1, M, N)

    # --- Alternative: fully fused Triton kernel (RMSNorm + RoPE + Attention) ---
    # from single_stream.engine.kernels.attention_fusion_ss import fused_rmsnorm_rope_attention_ss
    # k_norm = torch.empty((H, M, D), device=qkv.device, dtype=torch.float32)
    # o2 = torch.empty((M, N), device=qkv.device, dtype=torch.float32)
    # v = torch.empty((H, M, D), device=qkv.device, dtype=torch.float32)
    # fused_rmsnorm_rope_attention_ss[...](k_norm, o2, v, qkv, ...)
    # return o2.to(torch.bfloat16).view(1, M, N)


@fused_attention.register_fake
def _(qkv, q_norm_weight, k_norm_weight, rope_cos, rope_sin, n_sa):
    M = qkv.shape[0]
    return qkv.new_empty((1, M, 1536))
