"""RMSNorm + RoPE kernel for DoubleStreamBlock (Phase 1-3 only).

Outputs Q/K/V as (H, M, D) bf16, ready for F.scaled_dot_product_attention.
No attention computation — that is handled by the caller via F.sdpa.
"""

import triton
import triton.language as tl


@triton.jit
def rmsnorm_rope_kernel(
    # Output buffers (H, M, D) bf16
    Q_out_ptr,
    Q_out_stride0: tl.constexpr,
    Q_out_stride1: tl.constexpr,
    Q_out_stride2: tl.constexpr,
    K_out_ptr,
    K_out_stride0: tl.constexpr,
    K_out_stride1: tl.constexpr,
    K_out_stride2: tl.constexpr,
    V_out_ptr,
    V_out_stride0: tl.constexpr,
    V_out_stride1: tl.constexpr,
    V_out_stride2: tl.constexpr,
    # Input QKV
    SA_QKV_ptr,
    SA_QKV_stride0: tl.constexpr,
    SA_QKV_stride1: tl.constexpr,
    VL_QKV_ptr,
    VL_QKV_stride0: tl.constexpr,
    VL_QKV_stride1: tl.constexpr,
    # RMSNorm weights
    Q_norm_sa_ptr,
    K_norm_sa_ptr,
    Q_norm_vl_ptr,
    K_norm_vl_ptr,
    # RoPE tables
    ROPE_COS_ptr,
    ROPE_COS_stride0: tl.constexpr,
    ROPE_COS_stride1: tl.constexpr,
    ROPE_SIN_ptr,
    ROPE_SIN_stride0: tl.constexpr,
    ROPE_SIN_stride1: tl.constexpr,
    # Constants
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    N_VL: tl.constexpr,
    N_SA: tl.constexpr,
):
    """RMSNorm + RoPE only (Phase 1-3). Outputs Q/K/V as (H, M, D) bf16 for F.sdpa."""
    s = tl.program_id(0) * BLOCK_S
    h = tl.program_id(1)
    rs = s + tl.arange(0, BLOCK_S)
    rd = tl.arange(0, BLOCK_N)
    q_col = h * D
    k_col = N + h * D
    v_col = 2 * N + h * D
    mask_v = (rs[:, None] < M) & (rd[None, :] < D)
    is_vl = rs < N_VL
    is_sa = ~is_vl & (rs < M)
    vl_idx = rs
    sa_idx = tl.where(is_sa, rs - N_VL, 0)

    # Phase 1: Load QKV
    Q_vl = tl.load(
        VL_QKV_ptr + vl_idx[:, None] * VL_QKV_stride0 + (q_col + rd)[None, :] * VL_QKV_stride1,
        mask=is_vl[:, None] & (rd[None, :] < D),
        other=0.0,
    ).to(tl.float32)
    K_vl = tl.load(
        VL_QKV_ptr + vl_idx[:, None] * VL_QKV_stride0 + (k_col + rd)[None, :] * VL_QKV_stride1,
        mask=is_vl[:, None] & (rd[None, :] < D),
        other=0.0,
    ).to(tl.float32)
    V_vl = tl.load(
        VL_QKV_ptr + vl_idx[:, None] * VL_QKV_stride0 + (v_col + rd)[None, :] * VL_QKV_stride1,
        mask=is_vl[:, None] & (rd[None, :] < D),
        other=0.0,
    ).to(tl.float32)
    Q_sa = tl.load(
        SA_QKV_ptr + sa_idx[:, None] * SA_QKV_stride0 + (q_col + rd)[None, :] * SA_QKV_stride1,
        mask=is_sa[:, None] & (rd[None, :] < D),
        other=0.0,
    ).to(tl.float32)
    K_sa = tl.load(
        SA_QKV_ptr + sa_idx[:, None] * SA_QKV_stride0 + (k_col + rd)[None, :] * SA_QKV_stride1,
        mask=is_sa[:, None] & (rd[None, :] < D),
        other=0.0,
    ).to(tl.float32)
    V_sa = tl.load(
        SA_QKV_ptr + sa_idx[:, None] * SA_QKV_stride0 + (v_col + rd)[None, :] * SA_QKV_stride1,
        mask=is_sa[:, None] & (rd[None, :] < D),
        other=0.0,
    ).to(tl.float32)
    Q1 = tl.where(is_vl[:, None], Q_vl, Q_sa)
    K1 = tl.where(is_vl[:, None], K_vl, K_sa)
    V1 = tl.where(is_vl[:, None], V_vl, V_sa)

    # Phase 2: Store V (directly to bf16 output)
    tl.store(
        V_out_ptr + h * V_out_stride0 + rs[:, None] * V_out_stride1 + rd[None, :] * V_out_stride2,
        V1.to(tl.bfloat16),
        mask=mask_v,
    )

    # Phase 3: RMSNorm
    eps = 1e-6
    q_weight_sa = tl.load(Q_norm_sa_ptr + rd, mask=rd < D, other=1.0).to(tl.float32)
    k_weight_sa = tl.load(K_norm_sa_ptr + rd, mask=rd < D, other=1.0).to(tl.float32)
    q_weight_vl = tl.load(Q_norm_vl_ptr + rd, mask=rd < D, other=1.0).to(tl.float32)
    k_weight_vl = tl.load(K_norm_vl_ptr + rd, mask=rd < D, other=1.0).to(tl.float32)
    q_weight = tl.where(is_vl[:, None], q_weight_vl[None, :], q_weight_sa[None, :])
    k_weight = tl.where(is_vl[:, None], k_weight_vl[None, :], k_weight_sa[None, :])

    Q_rms_inv = tl.rsqrt(tl.sum(Q1 * Q1, axis=1) / D + eps)
    Q_pre = (Q1 * Q_rms_inv[:, None]).to(tl.bfloat16)
    Q_norm = Q_pre * q_weight.to(tl.bfloat16)  # bf16

    K_rms_inv = tl.rsqrt(tl.sum(K1 * K1, axis=1) / D + eps)
    K_pre = (K1 * K_rms_inv[:, None]).to(tl.bfloat16)
    K_norm_val = K_pre * k_weight.to(tl.bfloat16)  # bf16

    # Store un-RoPE'd Q/K
    tl.store(
        Q_out_ptr + h * Q_out_stride0 + rs[:, None] * Q_out_stride1 + rd[None, :] * Q_out_stride2,
        Q_norm,
        mask=mask_v,
    )
    tl.store(
        K_out_ptr + h * K_out_stride0 + rs[:, None] * K_out_stride1 + rd[None, :] * K_out_stride2,
        K_norm_val,
        mask=mask_v,
    )

    # Phase 3b: RoPE (SA only) — compute in fp32, store as bf16
    sa_start = M - N_SA
    is_sa_rope = rs >= sa_start
    any_sa = s + BLOCK_S > sa_start

    if any_sa:
        sa_rope_idx = tl.where(is_sa_rope, rs - sa_start, 0)
        rd2 = tl.arange(0, D // 2)
        re = 2 * rd2
        ro = re + 1
        mask_rope = (is_sa_rope & (rs < M))[:, None] & (rd2[None, :] < D // 2)

        cos = tl.load(
            ROPE_COS_ptr
            + sa_rope_idx[:, None] * ROPE_COS_stride0
            + rd2[None, :] * ROPE_COS_stride1,
            mask=mask_rope,
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            ROPE_SIN_ptr
            + sa_rope_idx[:, None] * ROPE_SIN_stride0
            + rd2[None, :] * ROPE_SIN_stride1,
            mask=mask_rope,
            other=0.0,
        ).to(tl.float32)

        q_e = tl.load(
            Q_out_ptr
            + h * Q_out_stride0
            + rs[:, None] * Q_out_stride1
            + re[None, :] * Q_out_stride2,
            mask=mask_rope,
            other=0.0,
        ).to(tl.float32)
        q_o = tl.load(
            Q_out_ptr
            + h * Q_out_stride0
            + rs[:, None] * Q_out_stride1
            + ro[None, :] * Q_out_stride2,
            mask=mask_rope,
            other=0.0,
        ).to(tl.float32)
        k_e = tl.load(
            K_out_ptr
            + h * K_out_stride0
            + rs[:, None] * K_out_stride1
            + re[None, :] * K_out_stride2,
            mask=mask_rope,
            other=0.0,
        ).to(tl.float32)
        k_o = tl.load(
            K_out_ptr
            + h * K_out_stride0
            + rs[:, None] * K_out_stride1
            + ro[None, :] * K_out_stride2,
            mask=mask_rope,
            other=0.0,
        ).to(tl.float32)

        tl.store(
            Q_out_ptr
            + h * Q_out_stride0
            + rs[:, None] * Q_out_stride1
            + re[None, :] * Q_out_stride2,
            (q_e * cos - q_o * sin).to(tl.bfloat16),
            mask=mask_rope,
        )
        tl.store(
            Q_out_ptr
            + h * Q_out_stride0
            + rs[:, None] * Q_out_stride1
            + ro[None, :] * Q_out_stride2,
            (q_e * sin + q_o * cos).to(tl.bfloat16),
            mask=mask_rope,
        )
        tl.store(
            K_out_ptr
            + h * K_out_stride0
            + rs[:, None] * K_out_stride1
            + re[None, :] * K_out_stride2,
            (k_e * cos - k_o * sin).to(tl.bfloat16),
            mask=mask_rope,
        )
        tl.store(
            K_out_ptr
            + h * K_out_stride0
            + rs[:, None] * K_out_stride1
            + ro[None, :] * K_out_stride2,
            (k_e * sin + k_o * cos).to(tl.bfloat16),
            mask=mask_rope,
        )
