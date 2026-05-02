"""RMSNorm + RoPE kernel for ExpandedSingleStreamBlock (3-way: [VL+SA | P]).

Extension of rmsnorm_rope_ss.py for 2-way joint attention.
Two separate QKV buffers: x_qkv (VL+SA) and p_qkv (P), each with own QK norms.

Layout in output: rows [0, N_x) = VL+SA, [N_x, M) = P
  - VL+SA: QK RMSNorm with x's weights, RoPE on SA portion only (axis0=0)
  - P: QK RMSNorm with p's weights, RoPE on all P (axis0=1)

Dtype: same as 2-way (fp32 RMSNorm, bf16 store, fp32 RoPE compute → bf16 store)
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
    # Input QKV — two separate buffers
    X_QKV_ptr,
    X_QKV_stride0: tl.constexpr,
    X_QKV_stride1: tl.constexpr,  # (N_x, QKV_DIM)
    P_QKV_ptr,
    P_QKV_stride0: tl.constexpr,
    P_QKV_stride1: tl.constexpr,  # (N_p, QKV_DIM)
    # RMSNorm weights — separate for x and p
    X_Q_norm_ptr,
    X_K_norm_ptr,
    P_Q_norm_ptr,
    P_K_norm_ptr,
    # RoPE tables — SA (axis0=0)
    SA_ROPE_COS_ptr,
    SA_ROPE_COS_stride0: tl.constexpr,
    SA_ROPE_COS_stride1: tl.constexpr,
    SA_ROPE_SIN_ptr,
    SA_ROPE_SIN_stride0: tl.constexpr,
    SA_ROPE_SIN_stride1: tl.constexpr,
    # RoPE tables — P (axis0=1)
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
    N_X: tl.constexpr,  # VL+SA tokens
    N_SA: tl.constexpr,  # SA portion (last N_SA of VL+SA)
    N_P: tl.constexpr,  # Physics tokens
):
    """RMSNorm + RoPE for [VL+SA | P]. Outputs Q/K/V as (H, M, D) bf16."""
    s = tl.program_id(0) * BLOCK_S
    h = tl.program_id(1)
    rs = s + tl.arange(0, BLOCK_S)
    rd = tl.arange(0, BLOCK_N)
    q_col = h * D
    k_col = N + h * D
    v_col = 2 * N + h * D
    mask_v = (rs[:, None] < M) & (rd[None, :] < D)

    # --- Region classification ---
    is_x = rs < N_X
    is_p = rs >= N_X
    x_idx = rs
    p_idx = tl.where(is_p, rs - N_X, 0)

    # --- Phase 1: Load QKV from two buffers ---
    Q_x = tl.load(
        X_QKV_ptr + x_idx[:, None] * X_QKV_stride0 + (q_col + rd)[None, :] * X_QKV_stride1,
        mask=is_x[:, None] & (rd[None, :] < D),
        other=0.0,
    ).to(tl.float32)
    K_x = tl.load(
        X_QKV_ptr + x_idx[:, None] * X_QKV_stride0 + (k_col + rd)[None, :] * X_QKV_stride1,
        mask=is_x[:, None] & (rd[None, :] < D),
        other=0.0,
    ).to(tl.float32)
    V_x = tl.load(
        X_QKV_ptr + x_idx[:, None] * X_QKV_stride0 + (v_col + rd)[None, :] * X_QKV_stride1,
        mask=is_x[:, None] & (rd[None, :] < D),
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

    Q1 = tl.where(is_x[:, None], Q_x, Q_p)
    K1 = tl.where(is_x[:, None], K_x, K_p)
    V1 = tl.where(is_x[:, None], V_x, V_p)

    # --- Phase 2: Store V ---
    tl.store(
        V_out_ptr + h * V_out_stride0 + rs[:, None] * V_out_stride1 + rd[None, :] * V_out_stride2,
        V1.to(tl.bfloat16),
        mask=mask_v,
    )

    # --- Phase 3: QK RMSNorm (separate weights for x vs p) ---
    eps = 1e-6
    x_q_w = tl.load(X_Q_norm_ptr + rd, mask=rd < D, other=1.0).to(tl.float32)
    x_k_w = tl.load(X_K_norm_ptr + rd, mask=rd < D, other=1.0).to(tl.float32)
    p_q_w = tl.load(P_Q_norm_ptr + rd, mask=rd < D, other=1.0).to(tl.float32)
    p_k_w = tl.load(P_K_norm_ptr + rd, mask=rd < D, other=1.0).to(tl.float32)

    q_weight = tl.where(is_x[:, None], x_q_w[None, :], p_q_w[None, :])
    k_weight = tl.where(is_x[:, None], x_k_w[None, :], p_k_w[None, :])

    Q_rms_inv = tl.rsqrt(tl.sum(Q1 * Q1, axis=1) / D + eps)
    Q_pre = (Q1 * Q_rms_inv[:, None]).to(tl.bfloat16)
    Q_norm = Q_pre * q_weight.to(tl.bfloat16)

    K_rms_inv = tl.rsqrt(tl.sum(K1 * K1, axis=1) / D + eps)
    K_pre = (K1 * K_rms_inv[:, None]).to(tl.bfloat16)
    K_norm_val = K_pre * k_weight.to(tl.bfloat16)

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

    # --- Phase 3b: RoPE (SA portion of x + all P) ---
    # SA RoPE (last N_SA of x region)
    sa_start = N_X - N_SA
    is_sa_rope = (rs >= sa_start) & (rs < N_X)
    any_sa = (s + BLOCK_S > sa_start) & (s < N_X)

    if any_sa:
        sa_rope_idx = tl.where(is_sa_rope, rs - sa_start, 0)
        rd2 = tl.arange(0, D // 2)
        re = 2 * rd2
        ro = re + 1
        mask_sa_rope = (is_sa_rope & (rs < M))[:, None] & (rd2[None, :] < D // 2)

        cos = tl.load(
            SA_ROPE_COS_ptr
            + sa_rope_idx[:, None] * SA_ROPE_COS_stride0
            + rd2[None, :] * SA_ROPE_COS_stride1,
            mask=mask_sa_rope,
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
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
            (q_e * cos - q_o * sin).to(tl.bfloat16),
            mask=mask_sa_rope,
        )
        tl.store(
            Q_out_ptr
            + h * Q_out_stride0
            + rs[:, None] * Q_out_stride1
            + ro[None, :] * Q_out_stride2,
            (q_e * sin + q_o * cos).to(tl.bfloat16),
            mask=mask_sa_rope,
        )
        tl.store(
            K_out_ptr
            + h * K_out_stride0
            + rs[:, None] * K_out_stride1
            + re[None, :] * K_out_stride2,
            (k_e * cos - k_o * sin).to(tl.bfloat16),
            mask=mask_sa_rope,
        )
        tl.store(
            K_out_ptr
            + h * K_out_stride0
            + rs[:, None] * K_out_stride1
            + ro[None, :] * K_out_stride2,
            (k_e * sin + k_o * cos).to(tl.bfloat16),
            mask=mask_sa_rope,
        )

    # P RoPE (all P tokens, axis0=1)
    p_start = N_X
    is_p_rope = rs >= p_start
    any_p = s + BLOCK_S > p_start

    if any_p:
        p_rope_idx = tl.where(is_p_rope, rs - p_start, 0)
        rd2 = tl.arange(0, D // 2)
        re = 2 * rd2
        ro = re + 1
        mask_p_rope = (is_p_rope & (rs < M))[:, None] & (rd2[None, :] < D // 2)

        cos = tl.load(
            P_ROPE_COS_ptr
            + p_rope_idx[:, None] * P_ROPE_COS_stride0
            + rd2[None, :] * P_ROPE_COS_stride1,
            mask=mask_p_rope,
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
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
            (q_e * cos - q_o * sin).to(tl.bfloat16),
            mask=mask_p_rope,
        )
        tl.store(
            Q_out_ptr
            + h * Q_out_stride0
            + rs[:, None] * Q_out_stride1
            + ro[None, :] * Q_out_stride2,
            (q_e * sin + q_o * cos).to(tl.bfloat16),
            mask=mask_p_rope,
        )
        tl.store(
            K_out_ptr
            + h * K_out_stride0
            + rs[:, None] * K_out_stride1
            + re[None, :] * K_out_stride2,
            (k_e * cos - k_o * sin).to(tl.bfloat16),
            mask=mask_p_rope,
        )
        tl.store(
            K_out_ptr
            + h * K_out_stride0
            + rs[:, None] * K_out_stride1
            + ro[None, :] * K_out_stride2,
            (k_e * sin + k_o * cos).to(tl.bfloat16),
            mask=mask_p_rope,
        )
