"""
Fused RMSNorm + RoPE + Attention kernel for SingleStreamBlock.

QKV is loaded from the pre-computed linear1 output (buf4) — no redundant matmul.
RMSNorm correctly applies q_norm.weight / k_norm.weight.

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
def fused_rmsnorm_rope_attention_ss(
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
    QKV_ptr,  # (M, >=3*N) bf16 — linear1 output (buf4)
    QKV_stride0: tl.constexpr,
    QKV_stride1: tl.constexpr,
    Q_norm_weight_ptr,  # (D,) bf16 — q_norm.weight
    K_norm_weight_ptr,  # (D,) bf16 — k_norm.weight
    ROPE_COS_ptr,
    ROPE_COS_stride0: tl.constexpr,
    ROPE_COS_stride1: tl.constexpr,
    ROPE_SIN_ptr,
    ROPE_SIN_stride0: tl.constexpr,
    ROPE_SIN_stride1: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    N_SA: tl.constexpr,
):
    """
    Fused RMSNorm + RoPE + Attention (fp32 GMEM scratch).
    QKV loaded from pre-computed linear1 output — no redundant matmul.
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

    # Phase 1: Load QKV from pre-computed linear1 output (buf4)
    Q1 = tl.load(
        QKV_ptr + rs[:, None] * QKV_stride0 + (q_col + rd)[None, :] * QKV_stride1,
        mask=mask_v,
        other=0.0,
    ).to(tl.float32)

    K1 = tl.load(
        QKV_ptr + rs[:, None] * QKV_stride0 + (k_col + rd)[None, :] * QKV_stride1,
        mask=mask_v,
        other=0.0,
    ).to(tl.float32)

    V1 = tl.load(
        QKV_ptr + rs[:, None] * QKV_stride0 + (v_col + rd)[None, :] * QKV_stride1,
        mask=mask_v,
        other=0.0,
    ).to(tl.float32)

    # Phase 2: Store V  # CHECKME: fp32 GMEM store (eager has no intermediate V storage)
    tl.store(
        V_ptr + h * V_stride0 + rs[:, None] * V_stride1 + rd[None, :] * V_stride2,
        V1,
        mask=mask_v,
    )

    # Phase 3: RMSNorm with weight multiplication
    eps = 1e-6

    q_weight = tl.load(Q_norm_weight_ptr + rd, mask=rd < D, other=1.0).to(tl.float32)
    k_weight = tl.load(K_norm_weight_ptr + rd, mask=rd < D, other=1.0).to(tl.float32)

    Q_sq = Q1 * Q1
    Q_rms_inv = tl.rsqrt(tl.sum(Q_sq, axis=1) / D + eps)
    Q_pre = (Q1 * Q_rms_inv[:, None]).to(tl.bfloat16)  # match eager: x * rsqrt(...), .type_as(x)
    Q_norm = (Q_pre * q_weight[None, :].to(tl.bfloat16)).to(
        tl.float32
    )  # bf16*bf16→fp32 for scratch

    K_sq = K1 * K1
    K_rms_inv = tl.rsqrt(tl.sum(K_sq, axis=1) / D + eps)
    K_pre = (K1 * K_rms_inv[:, None]).to(tl.bfloat16)  # match eager: x * rsqrt(...), .type_as(x)
    K_norm_val = (K_pre * k_weight[None, :].to(tl.bfloat16)).to(
        tl.float32
    )  # bf16*bf16→fp32 for scratch

    # Phase 3b: Store Q/K -> reload even/odd -> RoPE -> store back
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
    is_sa = rs >= sa_start

    any_sa = s + BLOCK_S > sa_start
    if any_sa:
        sa_idx = tl.where(is_sa, rs - sa_start, 0)

        rd2 = tl.arange(0, D // 2)
        re = 2 * rd2
        ro = re + 1

        mask_rope = (is_sa & (rs < M))[:, None] & (rd2[None, :] < D // 2)

        cos = tl.load(
            ROPE_COS_ptr + sa_idx[:, None] * ROPE_COS_stride0 + rd2[None, :] * ROPE_COS_stride1,
            mask=mask_rope,
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            ROPE_SIN_ptr + sa_idx[:, None] * ROPE_SIN_stride0 + rd2[None, :] * ROPE_SIN_stride1,
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

    # Phase 4: Scaled Dot-Product Attention (FA2-matching)
    Q_tile = tl.load(
        O2_ptr + rs[:, None] * O2_stride0 + (q_col + rd)[None, :] * O2_stride1,
        mask=mask_v,
        other=0.0,
    ).to(tl.bfloat16)

    scale = tl.rsqrt(tl.cast(D, tl.float32))
    lse_i = tl.full((BLOCK_S,), float("-inf"), dtype=tl.float32)
    m_i = tl.full((BLOCK_S,), float("-inf"), dtype=tl.float32)
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

        # Unscaled dot product — scale applied inside exp (FA2 FMA fusion)
        qk = tl.dot(Q_tile, tl.trans(K_tile))
        qk = tl.where(rp[None, :] < M, qk, float("-inf"))

        # FA2 online softmax with LSE tracking
        m_ij = tl.maximum(tl.max(qk, 1) * scale, lse_i)
        p_ij = tl.exp(qk * scale - m_ij[:, None])
        l_ij = tl.sum(p_ij, 1)

        # Rescale accumulator (FA2 order: rescale then add)
        acc_scale = tl.exp(m_i - m_ij)
        O = O * acc_scale[:, None]

        V_tile = tl.load(
            V_ptr + h * V_stride0 + rp[:, None] * V_stride1 + rd[None, :] * V_stride2,
            mask=mask_k,
            other=0.0,
        ).to(tl.bfloat16)

        O += tl.dot(p_ij.to(tl.bfloat16), V_tile)

        # Update LSE
        m_i = m_ij
        lse_i = m_ij + tl.log(tl.exp(lse_i - m_ij) + l_ij)

    # Final normalize (FA2 style: multiply by 1/sum_exp)
    o_scale = tl.exp(m_i - lse_i)
    O = O * o_scale[:, None]

    # Phase 5: Store attention output to O2 (bf16)
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
    QKV,
    q_norm_weight,
    k_norm_weight,
    rope_cos,
    rope_sin,
    n_sa=18,
    apply_rope=True,
):
    """
    Launch fused_rmsnorm_rope_attention_ss (RMSNorm + RoPE + Attention).
    QKV is the pre-computed linear1 output containing Q, K, V columns.
    """
    M = QKV.shape[0]
    N = O2.shape[1]
    H = V.shape[0]
    D = V.shape[2]

    if not apply_rope:
        n_sa_kernel = 0
        if rope_cos is None:
            rope_cos = torch.ones((1, D // 2), device=QKV.device, dtype=torch.float32)
            rope_sin = torch.zeros((1, D // 2), device=QKV.device, dtype=torch.float32)
    else:
        n_sa_kernel = n_sa
        if rope_cos is None or rope_sin is None:
            raise ValueError("rope_cos and rope_sin must be provided when apply_rope=True")

    rope_cos = rope_cos.to(device=QKV.device).contiguous()
    rope_sin = rope_sin.to(device=QKV.device).contiguous()

    fused_rmsnorm_rope_attention_ss[lambda meta: ((M + meta["BLOCK_S"] - 1) // meta["BLOCK_S"], H)](
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
        QKV,
        QKV.stride(0),
        QKV.stride(1),
        q_norm_weight,
        k_norm_weight,
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
        N_SA=n_sa_kernel,
    )
