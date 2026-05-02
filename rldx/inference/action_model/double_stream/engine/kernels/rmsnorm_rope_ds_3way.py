"""RMSNorm + RoPE kernel for ExpandedDoubleStreamBlock (3-way: VL|SA|P).

Extension of rmsnorm_rope_ds.py for 3-way joint attention [VL | SA | P].
Outputs Q/K/V as (H, M, D) bf16, ready for F.scaled_dot_product_attention.

Layout: rows [0, N_VL) = VL, [N_VL, N_VL+N_SA) = SA, [N_VL+N_SA, M) = P
  - VL: QK RMSNorm with vl weights, no RoPE
  - SA: QK RMSNorm with sa weights, RoPE with sa cos/sin
  - P:  QK RMSNorm with p weights, RoPE with p cos/sin

Dtype matching eager:
  - QKV load: bf16 (from Linear output)
  - RMSNorm: fp32 variance/rsqrt, cast to bf16 before weight multiply (bf16 × bf16)
  - RoPE: fp32 compute, bf16 store
  - V: direct bf16 passthrough
"""

import triton
import triton.language as tl


@triton.jit
def rmsnorm_rope_kernel_3way(
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
    # Input QKV (3 separate buffers)
    SA_QKV_ptr,
    SA_QKV_stride0: tl.constexpr,
    SA_QKV_stride1: tl.constexpr,
    VL_QKV_ptr,
    VL_QKV_stride0: tl.constexpr,
    VL_QKV_stride1: tl.constexpr,
    P_QKV_ptr,
    P_QKV_stride0: tl.constexpr,
    P_QKV_stride1: tl.constexpr,
    # RMSNorm weights (6 buffers: Q/K for SA, VL, P)
    Q_norm_sa_ptr,
    K_norm_sa_ptr,
    Q_norm_vl_ptr,
    K_norm_vl_ptr,
    Q_norm_p_ptr,
    K_norm_p_ptr,
    # RoPE tables — SA
    SA_ROPE_COS_ptr,
    SA_ROPE_COS_stride0: tl.constexpr,
    SA_ROPE_COS_stride1: tl.constexpr,
    SA_ROPE_SIN_ptr,
    SA_ROPE_SIN_stride0: tl.constexpr,
    SA_ROPE_SIN_stride1: tl.constexpr,
    # RoPE tables — P
    P_ROPE_COS_ptr,
    P_ROPE_COS_stride0: tl.constexpr,
    P_ROPE_COS_stride1: tl.constexpr,
    P_ROPE_SIN_ptr,
    P_ROPE_SIN_stride0: tl.constexpr,
    P_ROPE_SIN_stride1: tl.constexpr,
    # Constants
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    N_VL: tl.constexpr,
    N_SA: tl.constexpr,
    N_P: tl.constexpr,
):
    """RMSNorm + RoPE for 3-way [VL | SA | P]. Outputs Q/K/V as (H, M, D) bf16."""
    s = tl.program_id(0) * BLOCK_S
    h = tl.program_id(1)
    rs = s + tl.arange(0, BLOCK_S)
    rd = tl.arange(0, BLOCK_N)
    q_col = h * D
    k_col = N + h * D
    v_col = 2 * N + h * D
    mask_v = (rs[:, None] < M) & (rd[None, :] < D)

    # --- Region classification ---
    is_vl = rs < N_VL
    is_sa = (rs >= N_VL) & (rs < N_VL + N_SA)
    is_p = rs >= N_VL + N_SA
    vl_idx = rs
    sa_idx = tl.where(is_sa, rs - N_VL, 0)
    p_idx = tl.where(is_p, rs - N_VL - N_SA, 0)

    # --- Phase 1: Load QKV from 3 separate buffers ---
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

    Q_p = tl.load(
        P_QKV_ptr + p_idx[:, None] * P_QKV_stride0 + (q_col + rd)[None, :] * P_QKV_stride1,
        mask=is_p[:, None] & (rd[None, :] < D),
        other=0.0,
    ).to(tl.float32)
    K_p = tl.load(
        P_QKV_ptr + p_idx[:, None] * P_QKV_stride0 + (k_col + rd)[None, :] * P_QKV_stride1,
        mask=is_p[:, None] & (rd[None, :] < D),
        other=0.0,
    ).to(tl.float32)
    V_p = tl.load(
        P_QKV_ptr + p_idx[:, None] * P_QKV_stride0 + (v_col + rd)[None, :] * P_QKV_stride1,
        mask=is_p[:, None] & (rd[None, :] < D),
        other=0.0,
    ).to(tl.float32)

    # Merge: 3-way select
    Q1 = tl.where(is_vl[:, None], Q_vl, tl.where(is_sa[:, None], Q_sa, Q_p))
    K1 = tl.where(is_vl[:, None], K_vl, tl.where(is_sa[:, None], K_sa, K_p))
    V1 = tl.where(is_vl[:, None], V_vl, tl.where(is_sa[:, None], V_sa, V_p))

    # --- Phase 2: Store V (directly to bf16) ---
    tl.store(
        V_out_ptr + h * V_out_stride0 + rs[:, None] * V_out_stride1 + rd[None, :] * V_out_stride2,
        V1.to(tl.bfloat16),
        mask=mask_v,
    )

    # --- Phase 3: QK RMSNorm ---
    eps = 1e-6
    q_weight_sa = tl.load(Q_norm_sa_ptr + rd, mask=rd < D, other=1.0).to(tl.float32)
    k_weight_sa = tl.load(K_norm_sa_ptr + rd, mask=rd < D, other=1.0).to(tl.float32)
    q_weight_vl = tl.load(Q_norm_vl_ptr + rd, mask=rd < D, other=1.0).to(tl.float32)
    k_weight_vl = tl.load(K_norm_vl_ptr + rd, mask=rd < D, other=1.0).to(tl.float32)
    q_weight_p = tl.load(Q_norm_p_ptr + rd, mask=rd < D, other=1.0).to(tl.float32)
    k_weight_p = tl.load(K_norm_p_ptr + rd, mask=rd < D, other=1.0).to(tl.float32)

    # 3-way weight select
    q_weight = tl.where(
        is_vl[:, None],
        q_weight_vl[None, :],
        tl.where(is_sa[:, None], q_weight_sa[None, :], q_weight_p[None, :]),
    )
    k_weight = tl.where(
        is_vl[:, None],
        k_weight_vl[None, :],
        tl.where(is_sa[:, None], k_weight_sa[None, :], k_weight_p[None, :]),
    )

    # RMSNorm: fp32 variance/rsqrt → bf16 cast → weight multiply (bf16 × bf16)
    Q_rms_inv = tl.rsqrt(tl.sum(Q1 * Q1, axis=1) / D + eps)
    Q_pre = (Q1 * Q_rms_inv[:, None]).to(tl.bfloat16)
    Q_norm = Q_pre * q_weight.to(tl.bfloat16)

    K_rms_inv = tl.rsqrt(tl.sum(K1 * K1, axis=1) / D + eps)
    K_pre = (K1 * K_rms_inv[:, None]).to(tl.bfloat16)
    K_norm_val = K_pre * k_weight.to(tl.bfloat16)

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

    # --- Phase 3b: RoPE (SA + P regions) ---
    # SA RoPE
    sa_start = N_VL
    sa_end = N_VL + N_SA
    is_sa_rope = (rs >= sa_start) & (rs < sa_end)
    any_sa = (s + BLOCK_S > sa_start) & (s < sa_end)

    if any_sa:
        sa_rope_idx = tl.where(is_sa_rope, rs - sa_start, 0)
        rd2 = tl.arange(0, D // 2)
        re = 2 * rd2
        ro = re + 1
        mask_sa_rope = (is_sa_rope & (rs < M))[:, None] & (rd2[None, :] < D // 2)

        cos_sa = tl.load(
            SA_ROPE_COS_ptr
            + sa_rope_idx[:, None] * SA_ROPE_COS_stride0
            + rd2[None, :] * SA_ROPE_COS_stride1,
            mask=mask_sa_rope,
            other=1.0,
        ).to(tl.float32)
        sin_sa = tl.load(
            SA_ROPE_SIN_ptr
            + sa_rope_idx[:, None] * SA_ROPE_SIN_stride0
            + rd2[None, :] * SA_ROPE_SIN_stride1,
            mask=mask_sa_rope,
            other=0.0,
        ).to(tl.float32)

        q_e = tl.load(
            Q_out_ptr
            + h * Q_out_stride0
            + rs[:, None] * Q_out_stride1
            + re[None, :] * Q_out_stride2,
            mask=mask_sa_rope,
            other=0.0,
        ).to(tl.float32)
        q_o = tl.load(
            Q_out_ptr
            + h * Q_out_stride0
            + rs[:, None] * Q_out_stride1
            + ro[None, :] * Q_out_stride2,
            mask=mask_sa_rope,
            other=0.0,
        ).to(tl.float32)
        k_e = tl.load(
            K_out_ptr
            + h * K_out_stride0
            + rs[:, None] * K_out_stride1
            + re[None, :] * K_out_stride2,
            mask=mask_sa_rope,
            other=0.0,
        ).to(tl.float32)
        k_o = tl.load(
            K_out_ptr
            + h * K_out_stride0
            + rs[:, None] * K_out_stride1
            + ro[None, :] * K_out_stride2,
            mask=mask_sa_rope,
            other=0.0,
        ).to(tl.float32)

        tl.store(
            Q_out_ptr
            + h * Q_out_stride0
            + rs[:, None] * Q_out_stride1
            + re[None, :] * Q_out_stride2,
            (q_e * cos_sa - q_o * sin_sa).to(tl.bfloat16),
            mask=mask_sa_rope,
        )
        tl.store(
            Q_out_ptr
            + h * Q_out_stride0
            + rs[:, None] * Q_out_stride1
            + ro[None, :] * Q_out_stride2,
            (q_e * sin_sa + q_o * cos_sa).to(tl.bfloat16),
            mask=mask_sa_rope,
        )
        tl.store(
            K_out_ptr
            + h * K_out_stride0
            + rs[:, None] * K_out_stride1
            + re[None, :] * K_out_stride2,
            (k_e * cos_sa - k_o * sin_sa).to(tl.bfloat16),
            mask=mask_sa_rope,
        )
        tl.store(
            K_out_ptr
            + h * K_out_stride0
            + rs[:, None] * K_out_stride1
            + ro[None, :] * K_out_stride2,
            (k_e * sin_sa + k_o * cos_sa).to(tl.bfloat16),
            mask=mask_sa_rope,
        )

    # P RoPE
    p_start = N_VL + N_SA
    is_p_rope = rs >= p_start
    any_p = s + BLOCK_S > p_start

    if any_p:
        p_rope_idx = tl.where(is_p_rope, rs - p_start, 0)
        rd2 = tl.arange(0, D // 2)
        re = 2 * rd2
        ro = re + 1
        mask_p_rope = (is_p_rope & (rs < M))[:, None] & (rd2[None, :] < D // 2)

        cos_p = tl.load(
            P_ROPE_COS_ptr
            + p_rope_idx[:, None] * P_ROPE_COS_stride0
            + rd2[None, :] * P_ROPE_COS_stride1,
            mask=mask_p_rope,
            other=1.0,
        ).to(tl.float32)
        sin_p = tl.load(
            P_ROPE_SIN_ptr
            + p_rope_idx[:, None] * P_ROPE_SIN_stride0
            + rd2[None, :] * P_ROPE_SIN_stride1,
            mask=mask_p_rope,
            other=0.0,
        ).to(tl.float32)

        q_e = tl.load(
            Q_out_ptr
            + h * Q_out_stride0
            + rs[:, None] * Q_out_stride1
            + re[None, :] * Q_out_stride2,
            mask=mask_p_rope,
            other=0.0,
        ).to(tl.float32)
        q_o = tl.load(
            Q_out_ptr
            + h * Q_out_stride0
            + rs[:, None] * Q_out_stride1
            + ro[None, :] * Q_out_stride2,
            mask=mask_p_rope,
            other=0.0,
        ).to(tl.float32)
        k_e = tl.load(
            K_out_ptr
            + h * K_out_stride0
            + rs[:, None] * K_out_stride1
            + re[None, :] * K_out_stride2,
            mask=mask_p_rope,
            other=0.0,
        ).to(tl.float32)
        k_o = tl.load(
            K_out_ptr
            + h * K_out_stride0
            + rs[:, None] * K_out_stride1
            + ro[None, :] * K_out_stride2,
            mask=mask_p_rope,
            other=0.0,
        ).to(tl.float32)

        tl.store(
            Q_out_ptr
            + h * Q_out_stride0
            + rs[:, None] * Q_out_stride1
            + re[None, :] * Q_out_stride2,
            (q_e * cos_p - q_o * sin_p).to(tl.bfloat16),
            mask=mask_p_rope,
        )
        tl.store(
            Q_out_ptr
            + h * Q_out_stride0
            + rs[:, None] * Q_out_stride1
            + ro[None, :] * Q_out_stride2,
            (q_e * sin_p + q_o * cos_p).to(tl.bfloat16),
            mask=mask_p_rope,
        )
        tl.store(
            K_out_ptr
            + h * K_out_stride0
            + rs[:, None] * K_out_stride1
            + re[None, :] * K_out_stride2,
            (k_e * cos_p - k_o * sin_p).to(tl.bfloat16),
            mask=mask_p_rope,
        )
        tl.store(
            K_out_ptr
            + h * K_out_stride0
            + rs[:, None] * K_out_stride1
            + ro[None, :] * K_out_stride2,
            (k_e * sin_p + k_o * cos_p).to(tl.bfloat16),
            mask=mask_p_rope,
        )
