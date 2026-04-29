"""
Fused Vision RoPE + Non-causal Attention with Split-KV (Flash-Decoding).

Qwen3-VL Vision Encoder attention, reads fused QKV buffer directly with
baked cos/sin (rotate_half sign folded into rope_sin).

Invariants:
  - Per-seq 3D grid: program_id(2) = seq_idx (or packed split*seqs for the
    split kernel). No CTA spans a cu_seqlens boundary.
  - Safe-index clamping: masked-out lanes clamped to the last valid index.
    Triton masks gate the load *result*, not necessarily the memory
    transaction (Blackwell sm_120 enforces strictly).
  - QKV column layout at row m:
        [0, Q_DIM) = Q, [Q_DIM, 2*Q_DIM) = K, [2*Q_DIM, 3*Q_DIM) = V
    Head h, dim d at column (bank_offset) + h*D + d.

Split-KV dispatch (see SPLIT_M_THRESHOLD):
  M >= threshold -> direct kernel writes final O.
  M <  threshold -> split kernel writes partial (m, l, O) for NUM_SPLITS
    slices; reduce kernel merges. Partials are over-allocated to MAX_SPLITS
    with m = -inf pre-fill so unused slots contribute 0.
"""

import torch
import triton
import triton.language as tl


# Over-allocation ceiling for partial buffers. Autotune picks actual
# NUM_SPLITS <= this. Static ceiling keeps the reduce kernel's grid stable.
MAX_SPLITS = 8

# Direct is launched when M >= threshold; split otherwise. Tuned at build.
SPLIT_M_THRESHOLD = 128
THRESHOLD_CANDIDATES = (64, 128, 256, 512)


# ---------------------------------------------------------------------------
# Triton helpers — inlined at compile time (no runtime function boundary).
# Q_tile produced by _fused_vision_attention_q_rope flows as an SSA value into
# _fused_vision_attention_softmax; Triton never round-trips it through GMEM.  The split
# into two helpers is purely source-level organization.
# ---------------------------------------------------------------------------


@triton.jit
def _fused_vision_attention_q_rope(
    QKV_ptr,
    rope_cos_ptr,
    rope_sin_ptr,
    rs_safe,
    rd_safe,
    rd_rot_safe,
    qd_mask,
    q_col,
    D: tl.constexpr,
    QKV_DIM: tl.constexpr,
):
    """Load one (BLOCK_S, BLOCK_D) Q tile with baked RoPE applied."""
    q_self = tl.load(
        QKV_ptr + rs_safe[:, None] * QKV_DIM + (q_col + rd_safe)[None, :],
        mask=qd_mask,
        other=0.0,
    ).to(tl.float32)
    q_cross = tl.load(
        QKV_ptr + rs_safe[:, None] * QKV_DIM + (q_col + rd_rot_safe)[None, :],
        mask=qd_mask,
        other=0.0,
    ).to(tl.float32)
    rc = tl.load(
        rope_cos_ptr + rs_safe[:, None] * D + rd_safe[None, :],
        mask=qd_mask,
        other=1.0,
    ).to(tl.float32)
    rs_baked = tl.load(
        rope_sin_ptr + rs_safe[:, None] * D + rd_safe[None, :],
        mask=qd_mask,
        other=0.0,
    ).to(tl.float32)
    return (q_self * rc + q_cross * rs_baked).to(tl.bfloat16)


@triton.jit
def _fused_vision_attention_softmax(
    Q_tile,
    QKV_ptr,
    rope_cos_ptr,
    rope_sin_ptr,
    kv_lo,
    kv_hi,
    rd_safe,
    rd_rot_safe,
    d_mask,
    s_mask,
    scale,
    k_col,
    v_col,
    D: tl.constexpr,
    QKV_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Flash-attention online softmax over K/V rows in [kv_lo, kv_hi).

    Q_tile arrives as an SSA value (stays in registers/SMEM for the whole
    loop).  Returns (m, l, O_acc).
    """
    m_prev = tl.full([BLOCK_S], float("-inf"), dtype=tl.float32)
    l_prev = tl.zeros([BLOCK_S], dtype=tl.float32)
    O_acc = tl.zeros([BLOCK_S, BLOCK_D], dtype=tl.float32)

    for p_start in range(kv_lo, kv_hi, BLOCK_P):
        rp = p_start + tl.arange(0, BLOCK_P)
        p_mask = rp < kv_hi
        kd_mask = p_mask[:, None] & d_mask[None, :]
        rp_safe = tl.where(p_mask, rp, kv_hi - 1)

        k_self = tl.load(
            QKV_ptr + rp_safe[:, None] * QKV_DIM + (k_col + rd_safe)[None, :],
            mask=kd_mask,
            other=0.0,
        ).to(tl.float32)
        k_cross = tl.load(
            QKV_ptr + rp_safe[:, None] * QKV_DIM + (k_col + rd_rot_safe)[None, :],
            mask=kd_mask,
            other=0.0,
        ).to(tl.float32)
        kc = tl.load(
            rope_cos_ptr + rp_safe[:, None] * D + rd_safe[None, :],
            mask=kd_mask,
            other=1.0,
        ).to(tl.float32)
        ks_baked = tl.load(
            rope_sin_ptr + rp_safe[:, None] * D + rd_safe[None, :],
            mask=kd_mask,
            other=0.0,
        ).to(tl.float32)
        K_tile = (k_self * kc + k_cross * ks_baked).to(tl.bfloat16)

        S = tl.dot(Q_tile, tl.trans(K_tile)).to(tl.float32) * scale
        S = tl.where(s_mask[:, None] & p_mask[None, :], S, float("-inf"))

        m_cur = tl.max(S, axis=1)
        m_new = tl.maximum(m_prev, m_cur)
        alpha = tl.where(m_prev > float("-inf"), tl.exp(m_prev - m_new), 0.0)
        P = tl.where(m_new[:, None] > float("-inf"), tl.exp(S - m_new[:, None]), 0.0)
        l_new = alpha * l_prev + tl.sum(P, axis=1)

        V_tile = tl.load(
            QKV_ptr + rp_safe[:, None] * QKV_DIM + (v_col + rd_safe)[None, :],
            mask=kd_mask,
            other=0.0,
        )
        O_acc = O_acc * alpha[:, None] + tl.dot(P.to(tl.bfloat16), V_tile)

        m_prev = m_new
        l_prev = l_new

    return m_prev, l_prev, O_acc


# ---------------------------------------------------------------------------
# Split kernel (NUM_SPLITS > 1 path; writes partial (m, l, O))
# ---------------------------------------------------------------------------
# Grid: (cdiv(SEQ_LEN, BLOCK_S), NUM_HEADS, NUM_SPLITS * num_seqs)
# program_id(2) packs (split_idx, seq_idx); NUM_SPLITS is constexpr so divmod
# folds to constant ops.

_SPLIT_CONFIGS = [
    # NUM_SPLITS = 1 (kept for autotune comparability with direct)
    triton.Config({"BLOCK_S": 32, "BLOCK_P": 32, "NUM_SPLITS": 1}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_S": 32, "BLOCK_P": 64, "NUM_SPLITS": 1}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_S": 64, "BLOCK_P": 32, "NUM_SPLITS": 1}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_S": 64, "BLOCK_P": 64, "NUM_SPLITS": 1}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_S": 64, "BLOCK_P": 64, "NUM_SPLITS": 1}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_S": 128, "BLOCK_P": 64, "NUM_SPLITS": 1}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_S": 128, "BLOCK_P": 128, "NUM_SPLITS": 1}, num_stages=2, num_warps=8),
    # NUM_SPLITS = 2
    triton.Config({"BLOCK_S": 16, "BLOCK_P": 32, "NUM_SPLITS": 2}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_S": 32, "BLOCK_P": 32, "NUM_SPLITS": 2}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_S": 32, "BLOCK_P": 32, "NUM_SPLITS": 2}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_S": 32, "BLOCK_P": 64, "NUM_SPLITS": 2}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_S": 64, "BLOCK_P": 32, "NUM_SPLITS": 2}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_S": 64, "BLOCK_P": 64, "NUM_SPLITS": 2}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_S": 64, "BLOCK_P": 64, "NUM_SPLITS": 2}, num_stages=3, num_warps=8),
    # NUM_SPLITS = 4
    triton.Config({"BLOCK_S": 16, "BLOCK_P": 16, "NUM_SPLITS": 4}, num_stages=2, num_warps=2),
    triton.Config({"BLOCK_S": 16, "BLOCK_P": 32, "NUM_SPLITS": 4}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_S": 32, "BLOCK_P": 16, "NUM_SPLITS": 4}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_S": 32, "BLOCK_P": 32, "NUM_SPLITS": 4}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_S": 32, "BLOCK_P": 32, "NUM_SPLITS": 4}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_S": 32, "BLOCK_P": 64, "NUM_SPLITS": 4}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_S": 64, "BLOCK_P": 32, "NUM_SPLITS": 4}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_S": 64, "BLOCK_P": 64, "NUM_SPLITS": 4}, num_stages=2, num_warps=8),
    # NUM_SPLITS = 8 (for smallest M; maximum parallelism)
    triton.Config({"BLOCK_S": 16, "BLOCK_P": 16, "NUM_SPLITS": 8}, num_stages=2, num_warps=2),
    triton.Config({"BLOCK_S": 16, "BLOCK_P": 16, "NUM_SPLITS": 8}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_S": 16, "BLOCK_P": 32, "NUM_SPLITS": 8}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_S": 32, "BLOCK_P": 16, "NUM_SPLITS": 8}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_S": 32, "BLOCK_P": 32, "NUM_SPLITS": 8}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_S": 32, "BLOCK_P": 32, "NUM_SPLITS": 8}, num_stages=2, num_warps=8),
]


@triton.autotune(configs=_SPLIT_CONFIGS, key=["SEQ_LEN"])
@triton.jit
def fused_vision_attention_split_kernel(
    QKV_ptr,
    O_partial_ptr,  # (MAX_SPLITS, M, NUM_HEADS, D) bf16
    m_partial_ptr,  # (MAX_SPLITS, M, NUM_HEADS) fp32 (pre-filled -inf)
    l_partial_ptr,  # (MAX_SPLITS, M, NUM_HEADS) fp32
    rope_cos_ptr,
    rope_sin_ptr,
    cu_seqlens_ptr,  # (num_seqs+1,) int32
    SEQ_LEN,  # per-seq length; autotune key
    M,  # total rows (for partial strides)
    NUM_HEADS: tl.constexpr,
    D: tl.constexpr,
    Q_DIM: tl.constexpr,
    QKV_DIM: tl.constexpr,
    scale,
    NUM_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_s = tl.program_id(0)
    head = tl.program_id(1)
    pid_z = tl.program_id(2)
    split_idx = pid_z % NUM_SPLITS
    seq_idx = pid_z // NUM_SPLITS

    kv_start = tl.load(cu_seqlens_ptr + seq_idx)
    kv_end = tl.load(cu_seqlens_ptr + seq_idx + 1)

    # Per-split KV range (flash-decoding slicing).
    seq_len_i = kv_end - kv_start
    kv_lo = kv_start + (split_idx * seq_len_i) // NUM_SPLITS
    kv_hi = kv_start + ((split_idx + 1) * seq_len_i) // NUM_SPLITS

    # Indices + safe clamps.
    rs = kv_start + pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    rd = tl.arange(0, BLOCK_D)
    s_mask = rs < kv_end
    d_mask = rd < D
    qd_mask = s_mask[:, None] & d_mask[None, :]
    rs_safe = tl.where(s_mask, rs, kv_end - 1)
    HALF: tl.constexpr = D // 2
    rd_rot = tl.where(rd < HALF, rd + HALF, rd - HALF)
    rd_safe = tl.where(d_mask, rd, D - 1)
    rd_rot_safe = tl.where(d_mask, rd_rot, D - 1)

    q_col = head * D
    k_col = Q_DIM + head * D
    v_col = 2 * Q_DIM + head * D

    Q_tile = _fused_vision_attention_q_rope(
        QKV_ptr,
        rope_cos_ptr,
        rope_sin_ptr,
        rs_safe,
        rd_safe,
        rd_rot_safe,
        qd_mask,
        q_col,
        D,
        QKV_DIM,
    )

    m_final, l_final, O_acc = _fused_vision_attention_softmax(
        Q_tile,
        QKV_ptr,
        rope_cos_ptr,
        rope_sin_ptr,
        kv_lo,
        kv_hi,
        rd_safe,
        rd_rot_safe,
        d_mask,
        s_mask,
        scale,
        k_col,
        v_col,
        D,
        QKV_DIM,
        BLOCK_S,
        BLOCK_P,
        BLOCK_D,
    )

    # Store un-normalized O, m, l into partial buffers.
    o_base = (
        split_idx * M * NUM_HEADS * D
        + rs_safe[:, None] * (NUM_HEADS * D)
        + (head * D + rd_safe)[None, :]
    )
    tl.store(O_partial_ptr + o_base, O_acc.to(tl.bfloat16), mask=qd_mask)

    ml_base = split_idx * M * NUM_HEADS + rs_safe * NUM_HEADS + head
    tl.store(m_partial_ptr + ml_base, m_final, mask=s_mask)
    tl.store(l_partial_ptr + ml_base, l_final, mask=s_mask)


# ---------------------------------------------------------------------------
# Direct kernel (NUM_SPLITS = 1 fast path; writes final O)
# ---------------------------------------------------------------------------
# Grid: (cdiv(SEQ_LEN, BLOCK_S), NUM_HEADS, num_seqs)

_DIRECT_CONFIGS = [
    triton.Config({"BLOCK_S": 32, "BLOCK_P": 32}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_S": 32, "BLOCK_P": 64}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_S": 64, "BLOCK_P": 32}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_S": 64, "BLOCK_P": 64}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_S": 64, "BLOCK_P": 64}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_S": 128, "BLOCK_P": 64}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_S": 128, "BLOCK_P": 128}, num_stages=2, num_warps=8),
]


@triton.autotune(configs=_DIRECT_CONFIGS, key=["SEQ_LEN"])
@triton.jit
def fused_vision_attention_direct_kernel(
    QKV_ptr,
    O_ptr,  # (M, Q_DIM) bf16
    rope_cos_ptr,
    rope_sin_ptr,
    cu_seqlens_ptr,
    SEQ_LEN,  # per-seq length; autotune key
    NUM_HEADS: tl.constexpr,
    D: tl.constexpr,
    Q_DIM: tl.constexpr,
    QKV_DIM: tl.constexpr,
    scale,
    BLOCK_S: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_s = tl.program_id(0)
    head = tl.program_id(1)
    seq_idx = tl.program_id(2)

    kv_start = tl.load(cu_seqlens_ptr + seq_idx)
    kv_end = tl.load(cu_seqlens_ptr + seq_idx + 1)

    rs = kv_start + pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    rd = tl.arange(0, BLOCK_D)
    s_mask = rs < kv_end
    d_mask = rd < D
    qd_mask = s_mask[:, None] & d_mask[None, :]
    rs_safe = tl.where(s_mask, rs, kv_end - 1)
    HALF: tl.constexpr = D // 2
    rd_rot = tl.where(rd < HALF, rd + HALF, rd - HALF)
    rd_safe = tl.where(d_mask, rd, D - 1)
    rd_rot_safe = tl.where(d_mask, rd_rot, D - 1)

    q_col = head * D
    k_col = Q_DIM + head * D
    v_col = 2 * Q_DIM + head * D

    Q_tile = _fused_vision_attention_q_rope(
        QKV_ptr,
        rope_cos_ptr,
        rope_sin_ptr,
        rs_safe,
        rd_safe,
        rd_rot_safe,
        qd_mask,
        q_col,
        D,
        QKV_DIM,
    )

    _, l_final, O_acc = _fused_vision_attention_softmax(
        Q_tile,
        QKV_ptr,
        rope_cos_ptr,
        rope_sin_ptr,
        kv_start,
        kv_end,
        rd_safe,
        rd_rot_safe,
        d_mask,
        s_mask,
        scale,
        k_col,
        v_col,
        D,
        QKV_DIM,
        BLOCK_S,
        BLOCK_P,
        BLOCK_D,
    )

    l_safe = tl.where(l_final > 0.0, l_final, 1.0)
    O_final = O_acc / l_safe[:, None]
    tl.store(
        O_ptr + rs_safe[:, None] * Q_DIM + (head * D + rd_safe)[None, :],
        O_final.to(tl.bfloat16),
        mask=qd_mask,
    )


# ---------------------------------------------------------------------------
# Reduce kernel: merge MAX_SPLITS partials per (row, head) -> final O
# ---------------------------------------------------------------------------
# Unwritten split slots have m=-inf (from pre-fill) so alpha=0 below;
# l/O loads for those slots are mask-gated.
# Merge (flash-decoding reduction):
#   m_max = max_i(m_i)
#   a_i   = exp(m_i - m_max) (0 if m_i = -inf)
#   l_out = sum_i(a_i * l_i)
#   O_out = sum_i(a_i * O_i) / l_out


@triton.jit
def fused_vision_attention_reduce_kernel(
    O_partial_ptr,  # (MAX_SPLITS, M, NUM_HEADS, D) bf16
    m_partial_ptr,  # (MAX_SPLITS, M, NUM_HEADS) fp32
    l_partial_ptr,  # (MAX_SPLITS, M, NUM_HEADS) fp32
    O_ptr,  # (M, Q_DIM) bf16
    M,
    NUM_HEADS: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    D: tl.constexpr,
    Q_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_s = tl.program_id(0)
    head = tl.program_id(1)

    rs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    rd = tl.arange(0, BLOCK_D)
    s_mask = rs < M
    d_mask = rd < D
    rs_safe = tl.where(s_mask, rs, M - 1)
    rd_safe = tl.where(d_mask, rd, D - 1)

    # m_max per (row, head) across splits.
    rsplit = tl.arange(0, MAX_SPLITS)
    ml_offsets = rsplit[:, None] * (M * NUM_HEADS) + rs_safe[None, :] * NUM_HEADS + head
    m_all = tl.load(m_partial_ptr + ml_offsets, mask=s_mask[None, :], other=float("-inf"))
    m_max = tl.max(m_all, axis=0)

    valid_ml = (m_all > float("-inf")) & s_mask[None, :]
    alpha_all = tl.where(valid_ml, tl.exp(m_all - m_max[None, :]), 0.0)
    l_all = tl.load(l_partial_ptr + ml_offsets, mask=valid_ml, other=0.0)
    l_final = tl.sum(alpha_all * l_all, axis=0)

    O_acc = tl.zeros([BLOCK_S, BLOCK_D], dtype=tl.float32)
    for split in tl.static_range(MAX_SPLITS):
        m_val = tl.load(
            m_partial_ptr + split * M * NUM_HEADS + rs_safe * NUM_HEADS + head,
            mask=s_mask,
            other=float("-inf"),
        )
        valid_s = (m_val > float("-inf")) & s_mask
        a = tl.where(valid_s, tl.exp(m_val - m_max), 0.0)

        o_mask = valid_s[:, None] & d_mask[None, :]
        o_part = tl.load(
            O_partial_ptr
            + split * M * NUM_HEADS * D
            + rs_safe[:, None] * (NUM_HEADS * D)
            + (head * D + rd_safe)[None, :],
            mask=o_mask,
            other=0.0,
        ).to(tl.float32)

        O_acc += a[:, None] * o_part

    l_safe = tl.where(l_final > 0.0, l_final, 1.0)
    O_final = O_acc / l_safe[:, None]
    tl.store(
        O_ptr + rs_safe[:, None] * Q_DIM + (head * D + rd_safe)[None, :],
        O_final.to(tl.bfloat16),
        mask=s_mask[:, None] & d_mask[None, :],
    )


# ---------------------------------------------------------------------------
# Python API
# ---------------------------------------------------------------------------


def _run_direct(qkv, rope_cos, rope_sin, cu_seqlens, scaling, num_heads, head_dim, out):
    M = qkv.shape[0]
    num_seqs = cu_seqlens.shape[0] - 1
    seq_len = M // num_seqs  # uniform-seq assumption (RLDX fixed grid_thw)
    Q_DIM = num_heads * head_dim
    fused_vision_attention_direct_kernel[
        lambda meta: (triton.cdiv(seq_len, meta["BLOCK_S"]), num_heads, num_seqs)
    ](
        qkv,
        out,
        rope_cos,
        rope_sin,
        cu_seqlens,
        seq_len,
        NUM_HEADS=num_heads,
        D=head_dim,
        Q_DIM=Q_DIM,
        QKV_DIM=3 * Q_DIM,
        scale=scaling,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
    return out


def _run_split(qkv, rope_cos, rope_sin, cu_seqlens, scaling, num_heads, head_dim, out):
    M = qkv.shape[0]
    num_seqs = cu_seqlens.shape[0] - 1
    seq_len = M // num_seqs
    Q_DIM = num_heads * head_dim

    O_partial = torch.empty(
        (MAX_SPLITS, M, num_heads, head_dim),
        device=qkv.device,
        dtype=torch.bfloat16,
    )
    m_partial = torch.full(
        (MAX_SPLITS, M, num_heads),
        float("-inf"),
        device=qkv.device,
        dtype=torch.float32,
    )
    l_partial = torch.empty(
        (MAX_SPLITS, M, num_heads),
        device=qkv.device,
        dtype=torch.float32,
    )

    fused_vision_attention_split_kernel[
        lambda meta: (
            triton.cdiv(seq_len, meta["BLOCK_S"]),
            num_heads,
            meta["NUM_SPLITS"] * num_seqs,
        )
    ](
        qkv,
        O_partial,
        m_partial,
        l_partial,
        rope_cos,
        rope_sin,
        cu_seqlens,
        seq_len,
        M,
        NUM_HEADS=num_heads,
        D=head_dim,
        Q_DIM=Q_DIM,
        QKV_DIM=3 * Q_DIM,
        scale=scaling,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )

    BLOCK_S_R = 32
    BLOCK_D_R = triton.next_power_of_2(head_dim)
    fused_vision_attention_reduce_kernel[(triton.cdiv(M, BLOCK_S_R), num_heads)](
        O_partial,
        m_partial,
        l_partial,
        out,
        M,
        NUM_HEADS=num_heads,
        MAX_SPLITS=MAX_SPLITS,
        D=head_dim,
        Q_DIM=Q_DIM,
        BLOCK_S=BLOCK_S_R,
        BLOCK_D=BLOCK_D_R,
    )
    return out


def autotune_split_m_threshold(workloads, candidates=THRESHOLD_CANDIDATES, warmup=3, iters=10):
    """Pick the best direct-vs-split dispatch threshold for fixed workloads.

    Each workload: (qkv, rope_cos, rope_sin, cu_seqlens, scaling, num_heads, head_dim).
    """
    global SPLIT_M_THRESHOLD
    if not workloads:
        return SPLIT_M_THRESHOLD

    timings = {}
    for qkv, rope_cos, rope_sin, cu_seqlens, scaling, num_heads, head_dim in workloads:
        out = torch.empty(
            (qkv.shape[0], num_heads * head_dim),
            device=qkv.device,
            dtype=torch.bfloat16,
        )
        path_times = {}
        for name, fn in (("direct", _run_direct), ("split", _run_split)):
            for _ in range(warmup):
                fn(qkv, rope_cos, rope_sin, cu_seqlens, scaling, num_heads, head_dim, out)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn(qkv, rope_cos, rope_sin, cu_seqlens, scaling, num_heads, head_dim, out)
            end.record()
            torch.cuda.synchronize()
            path_times[name] = start.elapsed_time(end) / iters
        timings[qkv.shape[0]] = path_times

    best_threshold = SPLIT_M_THRESHOLD
    best_ms = None
    for threshold in candidates:
        total_ms = sum(
            timings[wl[0].shape[0]]["direct" if wl[0].shape[0] >= threshold else "split"]
            for wl in workloads
        )
        if best_ms is None or total_ms < best_ms:
            best_ms = total_ms
            best_threshold = threshold

    SPLIT_M_THRESHOLD = best_threshold
    return best_threshold


def forward(qkv, rope_cos, rope_sin, cu_seqlens, scaling, num_heads, head_dim):
    """Launch vision attention (direct or split-KV + reduce).

    Args:
        qkv:        (M, 3 * num_heads * head_dim) bf16
        rope_cos:   (M, head_dim) fp32
        rope_sin:   (M, head_dim) fp32 with rotate_half sign baked in
        cu_seqlens: (num_seqs+1,) int32
        scaling:    float
        num_heads:  int
        head_dim:   int

    Returns: (M, num_heads * head_dim) bf16
    """
    M = qkv.shape[0]
    rope_cos = rope_cos.contiguous()
    rope_sin = rope_sin.contiguous()
    O_out = torch.empty(
        (M, num_heads * head_dim),
        device=qkv.device,
        dtype=torch.bfloat16,
    )
    run = _run_direct if M >= SPLIT_M_THRESHOLD else _run_split
    return run(qkv, rope_cos, rope_sin, cu_seqlens, scaling, num_heads, head_dim, O_out)
