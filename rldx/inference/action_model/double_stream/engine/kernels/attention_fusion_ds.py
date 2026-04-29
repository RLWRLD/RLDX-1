"""
Fused RMSNorm + RoPE + Attention kernel for DoubleStreamBlock.

QKV is loaded from two pre-computed buffers:
  - SA_QKV (M_PADDED, 4608): SA tokens (bias already baked in by sa_ln_qkv_rmsnorm)
  - VL_QKV (N_VL, 4608): VL tokens (bias already baked in by vl_qkv linear)

RMSNorm uses dual weight pairs: (q_norm_sa, k_norm_sa) and (q_norm_vl, k_norm_vl).
RoPE is SA-only (identical to single stream), applied after RMSNorm.

fp32 GMEM scratch version — V, Q_norm (O2), K_norm stored in fp32,
cast to bf16 only for tensor core dot products (Phase 4).
"""

import torch
import triton
import triton.language as tl


def _build_rope_sa_only_tables(
    num_sa: int,
    head_dim: int,
    theta: float = 10000.0,
    axis0_dim: int = None,
    device=None,
    dtype=torch.bfloat16,
):
    """Build RoPE cos/sin tables for SA-only rope (bf16 output)."""
    if axis0_dim is None:
        axis0_dim = head_dim // 4
    axis1_dim = head_dim - axis0_dim

    assert axis0_dim % 2 == 0, f"axis0_dim must be even, got {axis0_dim}"
    assert axis1_dim % 2 == 0, f"axis1_dim must be even, got {axis1_dim}"

    pos = torch.arange(num_sa, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, axis1_dim, 2, device=device, dtype=torch.float32) / axis1_dim)
    )
    phase = torch.outer(pos, inv_freq)

    cos_axis1 = torch.cos(phase)
    sin_axis1 = torch.sin(phase)
    cos_axis0 = torch.ones((num_sa, axis0_dim // 2), device=device, dtype=torch.float32)
    sin_axis0 = torch.zeros((num_sa, axis0_dim // 2), device=device, dtype=torch.float32)

    cos_full = torch.cat([cos_axis0, cos_axis1], dim=1).contiguous()
    sin_full = torch.cat([sin_axis0, sin_axis1], dim=1).contiguous()

    return cos_full.to(dtype=dtype), sin_full.to(dtype=dtype)


@triton.autotune(
    # BLOCK_S >= 128 required: Phase 3b writes K/V, Phase 4 reads ALL K/V rows.
    # With BLOCK_S < M there are multiple blocks per head and no grid-wide barrier,
    # causing a race where early blocks read stale K/V from later blocks.
    configs=[
        triton.Config({"BLOCK_S": bs, "BLOCK_P": bp}, num_stages=ns, num_warps=nw)
        for bs in [128, 256, 512]
        for bp in [64, 128, 256]
        for ns in [1, 2, 3, 4]
        for nw in [4, 8, 16, 32]
    ],
    key=[],
)
@triton.jit
def fused_rmsnorm_rope_attention_ds(
    # Scratch buffers (fp32)
    K_norm_ptr,
    K_norm_stride0: tl.constexpr,
    K_norm_stride1: tl.constexpr,
    K_norm_stride2: tl.constexpr,
    O2_ptr,
    O2_stride0: tl.constexpr,
    O2_stride1: tl.constexpr,
    V_ptr,
    V_stride0: tl.constexpr,
    V_stride1: tl.constexpr,
    V_stride2: tl.constexpr,
    # Input QKV — two separate buffers (both with bias baked in)
    SA_QKV_ptr,  # (M_PADDED, 4608) bf16 — sa_ln_qkv_rmsnorm output
    SA_QKV_stride0: tl.constexpr,
    SA_QKV_stride1: tl.constexpr,
    VL_QKV_ptr,  # (N_VL, 4608) bf16 — VL QKV (bias baked in)
    VL_QKV_stride0: tl.constexpr,
    VL_QKV_stride1: tl.constexpr,
    # RMSNorm weights (4 vectors, each (D,))
    Q_norm_sa_ptr,  # q_norm_sa.weight (D,)
    K_norm_sa_ptr,  # k_norm_sa.weight (D,)
    Q_norm_vl_ptr,  # q_norm_vl.weight (D,)
    K_norm_vl_ptr,  # k_norm_vl.weight (D,)
    # RoPE tables (N_SA, D//2) float32
    ROPE_COS_ptr,
    ROPE_COS_stride0: tl.constexpr,
    ROPE_COS_stride1: tl.constexpr,
    ROPE_SIN_ptr,
    ROPE_SIN_stride0: tl.constexpr,
    ROPE_SIN_stride1: tl.constexpr,
    # Constants
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,  # TOTAL = N_VL + N_SA
    N: tl.constexpr,  # SA_DIM = 1536
    N_VL: tl.constexpr,
    N_SA: tl.constexpr,
):
    """
    Fused RMSNorm + RoPE + Attention for DoubleStreamBlock.

    Concatenated layout: rows [0, N_VL) = VL tokens, rows [N_VL, M) = SA tokens.
    QKV loaded from two separate pre-computed buffers (both with bias baked in).

    Grid: (ceil(M / BLOCK_S), H)
    """
    s = tl.program_id(0) * BLOCK_S
    h = tl.program_id(1)

    rs = s + tl.arange(0, BLOCK_S)
    rd = tl.arange(0, BLOCK_N)

    q_col = h * D
    k_col = N + h * D
    v_col = 2 * N + h * D

    mask_v = (rs[:, None] < M) & (rd[None, :] < D)

    # Row classification: VL rows [0, N_VL), SA rows [N_VL, M)
    is_vl = rs < N_VL
    is_sa = ~is_vl & (rs < M)

    # Indices into respective buffers
    vl_idx = rs  # valid when is_vl
    sa_idx = tl.where(is_sa, rs - N_VL, 0)  # valid when is_sa

    # ── Phase 1: Load QKV from two separate buffers ──────────────────────
    # Load VL QKV (bias already baked in)
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

    # Load SA QKV (bias already baked in)
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

    # Merge: VL rows get VL values, SA rows get SA values
    Q1 = tl.where(is_vl[:, None], Q_vl, Q_sa)
    K1 = tl.where(is_vl[:, None], K_vl, K_sa)
    V1 = tl.where(is_vl[:, None], V_vl, V_sa)

    # ── Phase 2: Store V ─────────────────────────────────────────────────
    tl.store(
        V_ptr + h * V_stride0 + rs[:, None] * V_stride1 + rd[None, :] * V_stride2,
        V1,
        mask=mask_v,  # CHECKME: fp32 GMEM store (eager has no intermediate V storage)
    )

    # ── Phase 3: RMSNorm with dual weight pairs ─────────────────────────
    eps = 1e-6

    # Load both sets of norm weights
    q_weight_sa = tl.load(Q_norm_sa_ptr + rd, mask=rd < D, other=1.0).to(tl.float32)
    k_weight_sa = tl.load(K_norm_sa_ptr + rd, mask=rd < D, other=1.0).to(tl.float32)
    q_weight_vl = tl.load(Q_norm_vl_ptr + rd, mask=rd < D, other=1.0).to(tl.float32)
    k_weight_vl = tl.load(K_norm_vl_ptr + rd, mask=rd < D, other=1.0).to(tl.float32)

    # Select weights based on row type
    q_weight = tl.where(is_vl[:, None], q_weight_vl[None, :], q_weight_sa[None, :])
    k_weight = tl.where(is_vl[:, None], k_weight_vl[None, :], k_weight_sa[None, :])

    Q_sq = Q1 * Q1
    Q_rms_inv = tl.rsqrt(tl.sum(Q_sq, axis=1) / D + eps)
    Q_pre = (Q1 * Q_rms_inv[:, None]).to(tl.bfloat16)  # match eager: x * rsqrt(...), .type_as(x)
    Q_norm = (Q_pre * q_weight.to(tl.bfloat16)).to(tl.float32)  # bf16*bf16→fp32 for scratch

    K_sq = K1 * K1
    K_rms_inv = tl.rsqrt(tl.sum(K_sq, axis=1) / D + eps)
    K_pre = (K1 * K_rms_inv[:, None]).to(tl.bfloat16)  # match eager: x * rsqrt(...), .type_as(x)
    K_norm_val = (K_pre * k_weight.to(tl.bfloat16)).to(tl.float32)  # bf16*bf16→fp32 for scratch

    # ── Phase 3b: Store Q to O2, K to K_norm → RoPE (SA-only) → store back
    tl.store(
        O2_ptr + rs[:, None] * O2_stride0 + (q_col + rd)[None, :] * O2_stride1,
        Q_norm,
        mask=mask_v,  # CHECKME: fp32
    )
    tl.store(
        K_norm_ptr
        + h * K_norm_stride0
        + rs[:, None] * K_norm_stride1
        + rd[None, :] * K_norm_stride2,
        K_norm_val,
        mask=mask_v,  # CHECKME: fp32
    )

    # Apply RoPE only to SA tokens (last N_SA rows)
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

        q_even = tl.load(
            O2_ptr + rs[:, None] * O2_stride0 + (q_col + re)[None, :] * O2_stride1,
            mask=mask_rope,
            other=0.0,
        ).to(tl.float32)
        q_odd = tl.load(
            O2_ptr + rs[:, None] * O2_stride0 + (q_col + ro)[None, :] * O2_stride1,
            mask=mask_rope,
            other=0.0,
        ).to(tl.float32)

        k_even = tl.load(
            K_norm_ptr
            + h * K_norm_stride0
            + rs[:, None] * K_norm_stride1
            + re[None, :] * K_norm_stride2,
            mask=mask_rope,
            other=0.0,
        ).to(tl.float32)
        k_odd = tl.load(
            K_norm_ptr
            + h * K_norm_stride0
            + rs[:, None] * K_norm_stride1
            + ro[None, :] * K_norm_stride2,
            mask=mask_rope,
            other=0.0,
        ).to(tl.float32)

        q_even_rot = q_even * cos - q_odd * sin
        q_odd_rot = q_even * sin + q_odd * cos
        k_even_rot = k_even * cos - k_odd * sin
        k_odd_rot = k_even * sin + k_odd * cos

        tl.store(
            O2_ptr + rs[:, None] * O2_stride0 + (q_col + re)[None, :] * O2_stride1,
            q_even_rot,
            mask=mask_rope,  # fp32
        )
        tl.store(
            O2_ptr + rs[:, None] * O2_stride0 + (q_col + ro)[None, :] * O2_stride1,
            q_odd_rot,
            mask=mask_rope,  # fp32
        )
        tl.store(
            K_norm_ptr
            + h * K_norm_stride0
            + rs[:, None] * K_norm_stride1
            + re[None, :] * K_norm_stride2,
            k_even_rot,
            mask=mask_rope,  # fp32
        )
        tl.store(
            K_norm_ptr
            + h * K_norm_stride0
            + rs[:, None] * K_norm_stride1
            + ro[None, :] * K_norm_stride2,
            k_odd_rot,
            mask=mask_rope,  # fp32
        )

    # ── Phase 4: Scaled Dot-Product Attention (VLM-matching, m/l separated) ──
    Q_tile = tl.load(
        O2_ptr + rs[:, None] * O2_stride0 + (q_col + rd)[None, :] * O2_stride1,
        mask=mask_v,
        other=0.0,
    ).to(tl.bfloat16)

    scale = tl.rsqrt(tl.cast(D, tl.float32))
    m_prev = tl.full((BLOCK_S,), float("-inf"), dtype=tl.float32)
    l_prev = tl.zeros((BLOCK_S,), dtype=tl.float32)
    O = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)

    for p in range(0, M, BLOCK_P):
        rp = p + tl.arange(0, BLOCK_P)

        mask_k = (rp[:, None] < M) & (rd[None, :] < D)
        K_tile = tl.load(
            K_norm_ptr
            + h * K_norm_stride0
            + rp[:, None] * K_norm_stride1
            + rd[None, :] * K_norm_stride2,
            mask=mask_k,
            other=0.0,
        ).to(tl.bfloat16)

        # Pre-scaled scores (VLM style)
        S = tl.dot(Q_tile, tl.trans(K_tile)).to(tl.float32) * scale
        S = tl.where(rp[None, :] < M, S, float("-inf"))

        # Online softmax: m/l separated (VLM style)
        m_cur = tl.max(S, axis=1)
        m_new = tl.maximum(m_prev, m_cur)
        alpha = tl.exp(m_prev - m_new)
        P = tl.exp(S - m_new[:, None])
        l_new = alpha * l_prev + tl.sum(P, axis=1)

        V_tile = tl.load(
            V_ptr + h * V_stride0 + rp[:, None] * V_stride1 + rd[None, :] * V_stride2,
            mask=mask_k,
            other=0.0,
        ).to(tl.bfloat16)

        O = O * alpha[:, None] + tl.dot(P.to(tl.bfloat16), V_tile)

        m_prev = m_new
        l_prev = l_new

    # Final normalize (VLM style: division)
    O = O / l_prev[:, None]

    # --- FA2-style (commented out for comparison) ---
    # lse_i = tl.full((BLOCK_S,), float('-inf'), dtype=tl.float32)
    # m_i = tl.full((BLOCK_S,), float('-inf'), dtype=tl.float32)
    # for p in range(0, M, BLOCK_P):
    #     qk = tl.dot(Q_tile, tl.trans(K_tile))
    #     m_ij = tl.maximum(tl.max(qk, 1) * scale, lse_i)
    #     p_ij = tl.exp(qk * scale - m_ij[:, None])
    #     l_ij = tl.sum(p_ij, 1)
    #     acc_scale = tl.exp(m_i - m_ij)
    #     O = O * acc_scale[:, None]
    #     O += tl.dot(p_ij.to(tl.bfloat16), V_tile)
    #     m_i = m_ij
    #     lse_i = m_ij + tl.log(tl.exp(lse_i - m_ij) + l_ij)
    # O = O * tl.exp(m_i - lse_i)[:, None]

    # ── Phase 5: Store attention output to O2 (bf16) ─────────────────────
    out_col = h * D
    mask_out = (rs[:, None] < M) & (rd[None, :] < D)
    tl.store(
        O2_ptr + rs[:, None] * O2_stride0 + (out_col + rd)[None, :] * O2_stride1,
        O.to(tl.bfloat16),
        mask=mask_out,
    )


def forward(
    K_norm,
    O2,
    V,
    SA_QKV,
    VL_QKV,
    q_norm_sa_weight,
    k_norm_sa_weight,
    q_norm_vl_weight,
    k_norm_vl_weight,
    rope_cos,
    rope_sin,
    n_sa,
    n_vl,
):
    """
    Launch fused_rmsnorm_rope_attention_ds (RMSNorm + RoPE + Attention) for DoubleStreamBlock.

    Args:
        K_norm: (H, TOTAL, D) fp32 scratch buffer
        O2:     (TOTAL, SA_DIM) fp32 scratch buffer
        V:      (H, TOTAL, D) fp32 scratch buffer
        SA_QKV: (M_PADDED, 4608) bf16 — SA QKV (bias baked in)
        VL_QKV: (N_VL, 4608) bf16 — VL QKV (bias baked in)
        q_norm_sa_weight, k_norm_sa_weight: (D,) bf16
        q_norm_vl_weight, k_norm_vl_weight: (D,) bf16
        rope_cos, rope_sin: (N_SA, D//2) float32
        n_sa: number of SA tokens
        n_vl: number of VL tokens
    """
    M = n_vl + n_sa
    N = O2.shape[1]  # SA_DIM = 1536
    H = V.shape[0]  # 24
    D = V.shape[2]  # 64

    rope_cos = rope_cos.contiguous()
    rope_sin = rope_sin.contiguous()

    fused_rmsnorm_rope_attention_ds[lambda meta: ((M + meta["BLOCK_S"] - 1) // meta["BLOCK_S"], H)](
        K_norm,
        K_norm.stride(0),
        K_norm.stride(1),
        K_norm.stride(2),
        O2,
        O2.stride(0),
        O2.stride(1),
        V,
        V.stride(0),
        V.stride(1),
        V.stride(2),
        SA_QKV,
        SA_QKV.stride(0),
        SA_QKV.stride(1),
        VL_QKV,
        VL_QKV.stride(0),
        VL_QKV.stride(1),
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
        BLOCK_N=D,
        D=D,
        H=H,
        M=M,
        N=N,
        N_VL=n_vl,
        N_SA=n_sa,
    )
