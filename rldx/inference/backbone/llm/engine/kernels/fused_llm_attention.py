"""
Fused RMSNorm + RoPE + Causal Attention with GQA grouping + Split-KV.

Single forward() fuses steps 2-7 of the VLM decoder layer:
  2. Fused QKV GEMM (caller)
  3. q_norm (RMSNorm) + weight + RoPE on Q
  5. k_norm (RMSNorm) + weight + RoPE on K (loaded ONCE per CTA, reused GROUP_SIZE×)
  6. V passthrough
  7. Causal online-softmax attention

GQA grouping:
  Grid iterates over NUM_KV_HEADS (not NUM_Q_HEADS).  Each CTA handles
  GROUP_SIZE Q heads sharing the same KV head, so K/V is loaded once
  and reused GROUP_SIZE× (4× on Qwen3: 32 Q / 8 KV).

  Q rows flatten to BS_EFF = BLOCK_S * GROUP_SIZE:
     row i → (seq_pos = rs[i // GROUP_SIZE], q_head = kv*G + i % G)

Split-KV (Flash-Decoding style):
  K/V range is divided into NUM_SPLITS slices; each CTA handles one slice,
  writes partial (m, l, O).  Reduction kernel merges per (row, q_head).
  Multiplies CTA count by NUM_SPLITS → more parallelism on small M.

  Causal + split: splits whose K range lies entirely past a row's q_idx
  produce (m=-inf, l=0, O=0) naturally; reduction gates these out via
  the `m_partial > -inf` check.

  NUM_SPLITS is chosen by @triton.autotune (keyed on M).

RoPE approach (eager-matching):
  Per-layer: q_norm_w, k_norm_w (+ rotated variants) — (D,) bf16
  Per-chain: cos, signed_sin — (M, D) bf16
  Weight applied first (matches eager q_norm output), then RoPE.

bf16 stores, fp32 RMSNorm / softmax compute.
Grid: (cdiv(M, BLOCK_S), NUM_KV_HEADS, NUM_SPLITS)

Partial buffer shapes (over-allocated to MAX_SPLITS for grid stability):
  O_partial: (MAX_SPLITS, M, NUM_Q_HEADS, D) bf16
  m_partial: (MAX_SPLITS, M, NUM_Q_HEADS) fp32 — pre-filled with -inf
  l_partial: (MAX_SPLITS, M, NUM_Q_HEADS) fp32 — unwritten slots guarded
"""

import torch
from torch._inductor.runtime.triton_helpers import libdevice
import triton
import triton.language as tl


# Max splits we ever allocate; autotune picks actual NUM_SPLITS <= this.
MAX_SPLITS = 8
THRESHOLD_CANDIDATES = (64, 128, 256, 512)

# M threshold: below this, use split kernel (more parallelism for small M);
# at/above, use direct kernel (skips partial alloc + reduce kernel overhead).
SPLIT_M_THRESHOLD = 128


# ---------------------------------------------------------------------------
# Split kernel — used when M < SPLIT_M_THRESHOLD (grid needs extra parallelism)
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        # BS_EFF = BLOCK_S * GROUP_SIZE (G=4 on Qwen3); kept small due to
        # register pressure of (BS_EFF, D) Q tile and (BS_EFF, D) O_acc.
        # NUM_SPLITS >= 2 only; NUM_SPLITS=1 is handled by the direct kernel.
        # --- NUM_SPLITS = 2 ---
        triton.Config({"BLOCK_S": 16, "BLOCK_P": 16, "NUM_SPLITS": 2}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_S": 16, "BLOCK_P": 16, "NUM_SPLITS": 2}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_S": 16, "BLOCK_P": 32, "NUM_SPLITS": 2}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_S": 16, "BLOCK_P": 32, "NUM_SPLITS": 2}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_S": 16, "BLOCK_P": 64, "NUM_SPLITS": 2}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_S": 16, "BLOCK_P": 64, "NUM_SPLITS": 2}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_S": 16, "BLOCK_P": 64, "NUM_SPLITS": 2}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_S": 32, "BLOCK_P": 32, "NUM_SPLITS": 2}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_S": 32, "BLOCK_P": 32, "NUM_SPLITS": 2}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_S": 32, "BLOCK_P": 64, "NUM_SPLITS": 2}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_S": 32, "BLOCK_P": 64, "NUM_SPLITS": 2}, num_stages=3, num_warps=8),
        # --- NUM_SPLITS = 4 ---
        triton.Config({"BLOCK_S": 16, "BLOCK_P": 16, "NUM_SPLITS": 4}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_S": 16, "BLOCK_P": 16, "NUM_SPLITS": 4}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_S": 16, "BLOCK_P": 32, "NUM_SPLITS": 4}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_S": 16, "BLOCK_P": 32, "NUM_SPLITS": 4}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_S": 16, "BLOCK_P": 32, "NUM_SPLITS": 4}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_S": 16, "BLOCK_P": 64, "NUM_SPLITS": 4}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_S": 16, "BLOCK_P": 64, "NUM_SPLITS": 4}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_S": 32, "BLOCK_P": 32, "NUM_SPLITS": 4}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_S": 32, "BLOCK_P": 32, "NUM_SPLITS": 4}, num_stages=2, num_warps=8),
        # --- NUM_SPLITS = 8 (smallest M) ---
        triton.Config({"BLOCK_S": 16, "BLOCK_P": 16, "NUM_SPLITS": 8}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_S": 16, "BLOCK_P": 16, "NUM_SPLITS": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_S": 16, "BLOCK_P": 32, "NUM_SPLITS": 8}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_S": 16, "BLOCK_P": 32, "NUM_SPLITS": 8}, num_stages=2, num_warps=4),
    ],
    key=["M"],
)
@triton.jit
def fused_llm_attention_split_kernel(
    # --- pointers ---
    QKV_ptr,  # (M, QKV_DIM) bf16
    O_partial_ptr,  # (MAX_SPLITS, M, NUM_Q_HEADS, D) bf16
    m_partial_ptr,  # (MAX_SPLITS, M, NUM_Q_HEADS) fp32 (pre-filled -inf)
    l_partial_ptr,  # (MAX_SPLITS, M, NUM_Q_HEADS) fp32
    q_norm_w_ptr,  # (D,) bf16 — per-layer
    q_norm_w_rot_ptr,  # (D,) bf16 — per-layer
    k_norm_w_ptr,  # (D,) bf16 — per-layer
    k_norm_w_rot_ptr,  # (D,) bf16 — per-layer
    cos_ptr,  # (M, D) bf16 — shared RoPE cos
    signed_sin_ptr,  # (M, D) bf16 — shared signed sin (rotate_half folded in)
    # --- runtime dims ---
    M,
    # --- constexpr model dims ---
    QKV_DIM: tl.constexpr,
    Q_DIM: tl.constexpr,
    K_DIM: tl.constexpr,
    D: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    # --- autotune constexprs ---
    NUM_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_s = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_idx = tl.program_id(2)

    # Flat (BS_EFF,) row layout: row i = (s=i//G, g=i%G)
    BS_EFF: tl.constexpr = BLOCK_S * GROUP_SIZE
    s_start = pid_s * BLOCK_S

    ri = tl.arange(0, BS_EFF)
    rs_flat = s_start + ri // GROUP_SIZE  # (BS_EFF,) seq position
    qh_flat = kv_head * GROUP_SIZE + (ri % GROUP_SIZE)  # (BS_EFF,) global Q head

    rd = tl.arange(0, BLOCK_D)
    d_mask = rd < D
    s_mask_flat = rs_flat < M
    qd_mask = s_mask_flat[:, None] & d_mask[None, :]

    HALF: tl.constexpr = D // 2
    rd_rot = tl.where(rd < HALF, rd + HALF, rd - HALF)

    eps: tl.constexpr = 1e-6
    D_float: tl.constexpr = D * 1.0
    scale = 1.0 / tl.sqrt(D_float)

    # This split's K range within the single contiguous sequence
    kv_start_split = (split_idx * M) // NUM_SPLITS
    kv_end_split = ((split_idx + 1) * M) // NUM_SPLITS

    # Norm weights (shared across all heads)
    qw = tl.load(q_norm_w_ptr + rd, mask=d_mask, other=0.0)
    qwr = tl.load(q_norm_w_rot_ptr + rd, mask=d_mask, other=0.0)
    kw = tl.load(k_norm_w_ptr + rd, mask=d_mask, other=0.0)
    kwr = tl.load(k_norm_w_rot_ptr + rd, mask=d_mask, other=0.0)

    # ================================================================
    # Phase 1: Q load + RMSNorm + weight + RoPE
    # ================================================================
    q_row = rs_flat[:, None] * QKV_DIM
    q_col_base = qh_flat[:, None] * D

    q_self = tl.load(
        QKV_ptr + q_row + q_col_base + rd[None, :],
        mask=qd_mask,
        other=0.0,
    ).to(tl.float32)

    q_cross = tl.load(
        QKV_ptr + q_row + q_col_base + rd_rot[None, :],
        mask=qd_mask,
        other=0.0,
    ).to(tl.float32)

    q_sq_sum = tl.sum(q_self * q_self, axis=1)
    q_rms_inv = libdevice.rsqrt(q_sq_sum / D_float + eps)

    q_scaled = (q_self * q_rms_inv[:, None]).to(tl.bfloat16)
    q_cross_scaled = (q_cross * q_rms_inv[:, None]).to(tl.bfloat16)

    q_normed = q_scaled * qw[None, :]
    q_cross_normed = q_cross_scaled * qwr[None, :]

    # RoPE: cos/sin depend only on seq_pos (rs_flat); redundant G× load
    # across group rows — a small inefficiency traded for code simplicity.
    cos_q = tl.load(cos_ptr + rs_flat[:, None] * D + rd[None, :], mask=qd_mask, other=0.0)
    ssin_q = tl.load(signed_sin_ptr + rs_flat[:, None] * D + rd[None, :], mask=qd_mask, other=0.0)

    Q_tile = q_normed * cos_q + q_cross_normed * ssin_q  # (BS_EFF, D) bf16

    # ================================================================
    # Phase 2: Online-softmax causal attention over this split's slice
    # ================================================================
    m_prev = tl.full([BS_EFF], float("-inf"), dtype=tl.float32)
    l_prev = tl.zeros([BS_EFF], dtype=tl.float32)
    O_acc = tl.zeros([BS_EFF, BLOCK_D], dtype=tl.float32)

    k_col = Q_DIM + kv_head * D  # single KV head per CTA
    v_col = Q_DIM + K_DIM + kv_head * D

    for p_start in range(kv_start_split, kv_end_split, BLOCK_P):
        rp = p_start + tl.arange(0, BLOCK_P)
        p_mask = rp < kv_end_split
        kd_mask = p_mask[:, None] & d_mask[None, :]

        # K load (BP, D) — shared across GROUP_SIZE Q heads
        k_self = tl.load(
            QKV_ptr + rp[:, None] * QKV_DIM + (k_col + rd)[None, :],
            mask=kd_mask,
            other=0.0,
        ).to(tl.float32)

        k_cross = tl.load(
            QKV_ptr + rp[:, None] * QKV_DIM + (k_col + rd_rot)[None, :],
            mask=kd_mask,
            other=0.0,
        ).to(tl.float32)

        k_sq_sum = tl.sum(k_self * k_self, axis=1)
        k_rms_inv = libdevice.rsqrt(k_sq_sum / D_float + eps)

        k_scaled = (k_self * k_rms_inv[:, None]).to(tl.bfloat16)
        k_cross_scaled = (k_cross * k_rms_inv[:, None]).to(tl.bfloat16)

        k_normed = k_scaled * kw[None, :]
        k_cross_normed = k_cross_scaled * kwr[None, :]

        cos_k = tl.load(cos_ptr + rp[:, None] * D + rd[None, :], mask=kd_mask, other=0.0)
        ssin_k = tl.load(signed_sin_ptr + rp[:, None] * D + rd[None, :], mask=kd_mask, other=0.0)

        K_tile = k_normed * cos_k + k_cross_normed * ssin_k  # (BP, D) bf16

        # Scores (BS_EFF, BP)
        S = tl.dot(Q_tile, tl.trans(K_tile)).to(tl.float32) * scale

        # Causal mask uses per-row q_idx = rs_flat[i]
        causal = rs_flat[:, None] >= rp[None, :]
        valid = s_mask_flat[:, None] & p_mask[None, :]
        S = tl.where(causal & valid, S, float("-inf"))

        # Online softmax (split-safe: guard -inf → 0)
        m_cur = tl.max(S, axis=1)
        m_new = tl.maximum(m_prev, m_cur)
        alpha = tl.where(m_prev > float("-inf"), tl.exp(m_prev - m_new), 0.0)
        P = tl.where(m_new[:, None] > float("-inf"), tl.exp(S - m_new[:, None]), 0.0)
        l_new = alpha * l_prev + tl.sum(P, axis=1)

        # V load (shared across group)
        V_tile = tl.load(
            QKV_ptr + rp[:, None] * QKV_DIM + (v_col + rd)[None, :],
            mask=kd_mask,
            other=0.0,
        )

        O_acc = O_acc * alpha[:, None] + tl.dot(P.to(tl.bfloat16), V_tile)

        m_prev = m_new
        l_prev = l_new

    # ================================================================
    # Phase 3: Store un-normalized partials (reduce kernel normalizes)
    # ================================================================
    # O_partial: (MAX_SPLITS, M, NUM_Q_HEADS, D)
    o_base = (
        split_idx * M * NUM_Q_HEADS * D
        + rs_flat[:, None] * (NUM_Q_HEADS * D)
        + qh_flat[:, None] * D
        + rd[None, :]
    )
    tl.store(O_partial_ptr + o_base, O_acc.to(tl.bfloat16), mask=qd_mask)

    ml_base = split_idx * M * NUM_Q_HEADS + rs_flat * NUM_Q_HEADS + qh_flat
    tl.store(m_partial_ptr + ml_base, m_prev, mask=s_mask_flat)
    tl.store(l_partial_ptr + ml_base, l_prev, mask=s_mask_flat)


# ---------------------------------------------------------------------------
# Reduce kernel: merge MAX_SPLITS partials per (row, q_head) → final O
# ---------------------------------------------------------------------------
# Unwritten slots have m=-inf (pre-fill), mapped to alpha=0.
#   m_max  = max_i(m_i)
#   a_i    = exp(m_i - m_max)    (0 if m_i == -inf)
#   l_out  = sum_i(a_i * l_i)
#   O_out  = sum_i(a_i * O_i) / l_out


@triton.jit
def fused_llm_attention_reduce_kernel(
    O_partial_ptr,  # (MAX_SPLITS, M, NUM_HEADS, D) bf16
    m_partial_ptr,  # (MAX_SPLITS, M, NUM_HEADS) fp32
    l_partial_ptr,  # (MAX_SPLITS, M, NUM_HEADS) fp32
    O_ptr,  # (M, Q_DIM) bf16
    M,
    NUM_HEADS: tl.constexpr,
    MAX_SPLITS_CE: tl.constexpr,
    D: tl.constexpr,
    Q_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_s = tl.program_id(0)
    head = tl.program_id(1)

    s_start = pid_s * BLOCK_S
    rs = s_start + tl.arange(0, BLOCK_S)
    rd = tl.arange(0, BLOCK_D)
    s_mask = rs < M
    d_mask = rd < D

    # Load all partial m → find m_max per row
    rsplit = tl.arange(0, MAX_SPLITS_CE)
    ml_offsets = rsplit[:, None] * (M * NUM_HEADS) + rs[None, :] * NUM_HEADS + head
    m_all = tl.load(m_partial_ptr + ml_offsets, mask=s_mask[None, :], other=float("-inf"))
    m_max = tl.max(m_all, axis=0)  # (BLOCK_S,)

    valid_ml = (m_all > float("-inf")) & s_mask[None, :]
    alpha_all = tl.where(valid_ml, tl.exp(m_all - m_max[None, :]), 0.0)
    l_all = tl.load(l_partial_ptr + ml_offsets, mask=valid_ml, other=0.0)
    l_final = tl.sum(alpha_all * l_all, axis=0)  # (BLOCK_S,)

    O_acc = tl.zeros([BLOCK_S, BLOCK_D], dtype=tl.float32)
    for split in tl.static_range(MAX_SPLITS_CE):
        m_val = tl.load(
            m_partial_ptr + split * M * NUM_HEADS + rs * NUM_HEADS + head,
            mask=s_mask,
            other=float("-inf"),
        )
        valid_s = (m_val > float("-inf")) & s_mask
        a = tl.where(valid_s, tl.exp(m_val - m_max), 0.0)

        o_mask = valid_s[:, None] & d_mask[None, :]
        o_part = tl.load(
            O_partial_ptr
            + split * M * NUM_HEADS * D
            + rs[:, None] * (NUM_HEADS * D)
            + (head * D + rd)[None, :],
            mask=o_mask,
            other=0.0,
        ).to(tl.float32)

        O_acc += a[:, None] * o_part

    l_safe = tl.where(l_final > 0.0, l_final, 1.0)
    O_final = O_acc / l_safe[:, None]

    tl.store(
        O_ptr + rs[:, None] * Q_DIM + (head * D + rd)[None, :],
        O_final.to(tl.bfloat16),
        mask=s_mask[:, None] & d_mask[None, :],
    )


# ---------------------------------------------------------------------------
# Direct kernel (NUM_SPLITS=1 fast path)
# ---------------------------------------------------------------------------
# Used when M is large enough that NUM_KV_HEADS × cdiv(M, BLOCK_S) already
# saturates the GPU.  Normalizes in-kernel and writes straight to O_out,
# skipping the partial buffer allocation and reduce kernel launch.
#
# Three LLM-causal-specific optimizations over the split kernel:
#   1. Causal early-exit: outer K loop stops at min(M, s_start + BLOCK_S)
#      since K beyond a Q tile's causal range is all masked → ~2× fewer
#      inner iterations on average.
#   2. Prefix / boundary split: tiles fully before s_start are in the
#      causal prefix (all K valid) — skip the (BS_EFF, BP) causal mask
#      generation in the hot path.
#   3. Group-shared RoPE: load cos/sin as (BLOCK_S, D) once and broadcast
#      via (BS, G, D) reshape — GROUP_SIZE=4× fewer cos/sin HBM reads
#      than naïve flat-indexing (which re-reads the same row per group).
#
# Grid: (cdiv(M, BLOCK_S), NUM_KV_HEADS)


@triton.autotune(
    configs=[
        # BLOCK_S=16 (BS_EFF=64 Q rows)
        triton.Config(
            {
                "BLOCK_S": 16,
                "BLOCK_P": 16,
            },
            num_stages=2,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_S": 16,
                "BLOCK_P": 16,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_S": 16,
                "BLOCK_P": 32,
            },
            num_stages=2,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_S": 16,
                "BLOCK_P": 32,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_S": 16,
                "BLOCK_P": 32,
            },
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_S": 16,
                "BLOCK_P": 64,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_S": 16,
                "BLOCK_P": 64,
            },
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_S": 16,
                "BLOCK_P": 64,
            },
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_S": 16,
                "BLOCK_P": 64,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_S": 16,
                "BLOCK_P": 128,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_S": 16,
                "BLOCK_P": 128,
            },
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_S": 16,
                "BLOCK_P": 128,
            },
            num_stages=3,
            num_warps=8,
        ),
        # BLOCK_S=32 (BS_EFF=128 Q rows) — bigger tiles for large M
        triton.Config(
            {
                "BLOCK_S": 32,
                "BLOCK_P": 16,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_S": 32,
                "BLOCK_P": 32,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_S": 32,
                "BLOCK_P": 32,
            },
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_S": 32,
                "BLOCK_P": 32,
            },
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_S": 32,
                "BLOCK_P": 32,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_S": 32,
                "BLOCK_P": 64,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_S": 32,
                "BLOCK_P": 64,
            },
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_S": 32,
                "BLOCK_P": 64,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_S": 32,
                "BLOCK_P": 128,
            },
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_S": 32,
                "BLOCK_P": 128,
            },
            num_stages=3,
            num_warps=8,
        ),
    ],
    key=["M"],
)
@triton.jit
def fused_llm_attention_direct_kernel(
    QKV_ptr,  # (M, QKV_DIM) bf16
    O_ptr,  # (M, Q_DIM) bf16 — written directly, normalized
    q_norm_w_ptr,
    q_norm_w_rot_ptr,
    k_norm_w_ptr,
    k_norm_w_rot_ptr,
    cos_ptr,
    signed_sin_ptr,
    M,
    QKV_DIM: tl.constexpr,
    Q_DIM: tl.constexpr,
    K_DIM: tl.constexpr,
    D: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_s = tl.program_id(0)
    kv_head = tl.program_id(1)

    BS_EFF: tl.constexpr = BLOCK_S * GROUP_SIZE
    s_start = pid_s * BLOCK_S

    # Flat (BS_EFF,) layout: row i = (seq_pos = s_start + i//G, q_head = kv*G + i%G)
    ri = tl.arange(0, BS_EFF)
    rs_flat = s_start + ri // GROUP_SIZE
    qh_flat = kv_head * GROUP_SIZE + (ri % GROUP_SIZE)

    # 1D (BS,) for group-shared cos/sin (4 Q heads per group share seq_pos)
    rs_bs = s_start + tl.arange(0, BLOCK_S)
    bs_mask = rs_bs < M

    rd = tl.arange(0, BLOCK_D)
    d_mask = rd < D
    s_mask_flat = rs_flat < M
    qd_mask = s_mask_flat[:, None] & d_mask[None, :]
    bsd_mask = bs_mask[:, None] & d_mask[None, :]

    HALF: tl.constexpr = D // 2
    rd_rot = tl.where(rd < HALF, rd + HALF, rd - HALF)

    eps: tl.constexpr = 1e-6
    D_float: tl.constexpr = D * 1.0
    scale = 1.0 / tl.sqrt(D_float)

    qw = tl.load(q_norm_w_ptr + rd, mask=d_mask, other=0.0)
    qwr = tl.load(q_norm_w_rot_ptr + rd, mask=d_mask, other=0.0)
    kw = tl.load(k_norm_w_ptr + rd, mask=d_mask, other=0.0)
    kwr = tl.load(k_norm_w_rot_ptr + rd, mask=d_mask, other=0.0)

    # ----- Q load + RMSNorm + weight -----
    q_row = rs_flat[:, None] * QKV_DIM
    q_col_base = qh_flat[:, None] * D

    q_self = tl.load(QKV_ptr + q_row + q_col_base + rd[None, :], mask=qd_mask, other=0.0).to(
        tl.float32
    )
    q_cross = tl.load(QKV_ptr + q_row + q_col_base + rd_rot[None, :], mask=qd_mask, other=0.0).to(
        tl.float32
    )

    q_sq_sum = tl.sum(q_self * q_self, axis=1)
    q_rms_inv = libdevice.rsqrt(q_sq_sum / D_float + eps)
    q_scaled = (q_self * q_rms_inv[:, None]).to(tl.bfloat16)
    q_cross_scaled = (q_cross * q_rms_inv[:, None]).to(tl.bfloat16)

    q_normed = q_scaled * qw[None, :]  # (BS_EFF, D) bf16
    q_cross_normed = q_cross_scaled * qwr[None, :]

    # ----- RoPE via group-shared cos/sin -----
    # G Q-heads in a group share seq_pos → cos/sin identical.  Load (BS, D)
    # once, broadcast across group via (BS, G, D) reshape.
    cos_bs = tl.load(
        cos_ptr + rs_bs[:, None] * D + rd[None, :], mask=bsd_mask, other=0.0
    )  # (BS, D) bf16
    ssin_bs = tl.load(signed_sin_ptr + rs_bs[:, None] * D + rd[None, :], mask=bsd_mask, other=0.0)

    q_normed_3d = tl.reshape(q_normed, [BLOCK_S, GROUP_SIZE, BLOCK_D])
    q_cross_3d = tl.reshape(q_cross_normed, [BLOCK_S, GROUP_SIZE, BLOCK_D])
    Q_3d = q_normed_3d * cos_bs[:, None, :] + q_cross_3d * ssin_bs[:, None, :]
    Q_tile = tl.reshape(Q_3d, [BS_EFF, BLOCK_D])  # (BS_EFF, D) bf16

    # ================================================================
    # Online-softmax causal attention
    #
    # Causal early-exit: valid K range is [0, s_start + BLOCK_S) since
    # rs_flat max = s_start + BLOCK_S - 1 (all K beyond are fully masked
    # and contribute nothing — skip them).
    #
    # Prefix / boundary split:
    #   prefix   = K tiles entirely before s_start  → causal always True,
    #              skip the (BS_EFF, BP) causal mask generation.
    #   boundary = K tiles overlapping [s_start, kv_end)  → causal mask.
    # ================================================================
    m_prev = tl.full([BS_EFF], float("-inf"), dtype=tl.float32)
    l_prev = tl.zeros([BS_EFF], dtype=tl.float32)
    O_acc = tl.zeros([BS_EFF, BLOCK_D], dtype=tl.float32)

    k_col = Q_DIM + kv_head * D
    v_col = Q_DIM + K_DIM + kv_head * D

    kv_end = tl.minimum(M, s_start + BLOCK_S)
    # Largest multiple of BLOCK_P ≤ s_start — end of fully-valid prefix region.
    prefix_end = (s_start // BLOCK_P) * BLOCK_P

    # ---- Prefix: causal always True (rp < s_start ≤ rs_flat) ----
    for p_start in range(0, prefix_end, BLOCK_P):
        rp = p_start + tl.arange(0, BLOCK_P)
        # p_mask: trivially True here (p_start + BP ≤ prefix_end ≤ s_start < M),
        # kept for uniform 2D mask shape.
        p_mask = rp < M
        kd_mask = p_mask[:, None] & d_mask[None, :]

        k_self = tl.load(
            QKV_ptr + rp[:, None] * QKV_DIM + (k_col + rd)[None, :], mask=kd_mask, other=0.0
        ).to(tl.float32)
        k_cross = tl.load(
            QKV_ptr + rp[:, None] * QKV_DIM + (k_col + rd_rot)[None, :], mask=kd_mask, other=0.0
        ).to(tl.float32)

        k_sq_sum = tl.sum(k_self * k_self, axis=1)
        k_rms_inv = libdevice.rsqrt(k_sq_sum / D_float + eps)
        k_scaled = (k_self * k_rms_inv[:, None]).to(tl.bfloat16)
        k_cross_scaled = (k_cross * k_rms_inv[:, None]).to(tl.bfloat16)

        k_normed = k_scaled * kw[None, :]
        k_cross_normed = k_cross_scaled * kwr[None, :]

        cos_k = tl.load(cos_ptr + rp[:, None] * D + rd[None, :], mask=kd_mask, other=0.0)
        ssin_k = tl.load(signed_sin_ptr + rp[:, None] * D + rd[None, :], mask=kd_mask, other=0.0)
        K_tile = k_normed * cos_k + k_cross_normed * ssin_k

        S = tl.dot(Q_tile, tl.trans(K_tile)).to(tl.float32) * scale
        # Causal omitted in prefix; mask padding rows only.
        S = tl.where(s_mask_flat[:, None], S, float("-inf"))

        m_cur = tl.max(S, axis=1)
        m_new = tl.maximum(m_prev, m_cur)
        alpha = tl.where(m_prev > float("-inf"), tl.exp(m_prev - m_new), 0.0)
        P = tl.where(m_new[:, None] > float("-inf"), tl.exp(S - m_new[:, None]), 0.0)
        l_new = alpha * l_prev + tl.sum(P, axis=1)

        V_tile = tl.load(
            QKV_ptr + rp[:, None] * QKV_DIM + (v_col + rd)[None, :], mask=kd_mask, other=0.0
        )
        O_acc = O_acc * alpha[:, None] + tl.dot(P.to(tl.bfloat16), V_tile)

        m_prev = m_new
        l_prev = l_new

    # ---- Boundary: causal mask needed (tile overlaps Q's range) ----
    for p_start in range(prefix_end, kv_end, BLOCK_P):
        rp = p_start + tl.arange(0, BLOCK_P)
        p_mask = rp < M
        kd_mask = p_mask[:, None] & d_mask[None, :]

        k_self = tl.load(
            QKV_ptr + rp[:, None] * QKV_DIM + (k_col + rd)[None, :], mask=kd_mask, other=0.0
        ).to(tl.float32)
        k_cross = tl.load(
            QKV_ptr + rp[:, None] * QKV_DIM + (k_col + rd_rot)[None, :], mask=kd_mask, other=0.0
        ).to(tl.float32)

        k_sq_sum = tl.sum(k_self * k_self, axis=1)
        k_rms_inv = libdevice.rsqrt(k_sq_sum / D_float + eps)
        k_scaled = (k_self * k_rms_inv[:, None]).to(tl.bfloat16)
        k_cross_scaled = (k_cross * k_rms_inv[:, None]).to(tl.bfloat16)

        k_normed = k_scaled * kw[None, :]
        k_cross_normed = k_cross_scaled * kwr[None, :]

        cos_k = tl.load(cos_ptr + rp[:, None] * D + rd[None, :], mask=kd_mask, other=0.0)
        ssin_k = tl.load(signed_sin_ptr + rp[:, None] * D + rd[None, :], mask=kd_mask, other=0.0)
        K_tile = k_normed * cos_k + k_cross_normed * ssin_k

        S = tl.dot(Q_tile, tl.trans(K_tile)).to(tl.float32) * scale

        causal = rs_flat[:, None] >= rp[None, :]
        valid = s_mask_flat[:, None] & p_mask[None, :]
        S = tl.where(causal & valid, S, float("-inf"))

        m_cur = tl.max(S, axis=1)
        m_new = tl.maximum(m_prev, m_cur)
        alpha = tl.where(m_prev > float("-inf"), tl.exp(m_prev - m_new), 0.0)
        P = tl.where(m_new[:, None] > float("-inf"), tl.exp(S - m_new[:, None]), 0.0)
        l_new = alpha * l_prev + tl.sum(P, axis=1)

        V_tile = tl.load(
            QKV_ptr + rp[:, None] * QKV_DIM + (v_col + rd)[None, :], mask=kd_mask, other=0.0
        )
        O_acc = O_acc * alpha[:, None] + tl.dot(P.to(tl.bfloat16), V_tile)

        m_prev = m_new
        l_prev = l_new

    # ----- Normalize and write directly to O_out -----
    # Guard against l_prev == 0 (pure-padding row; result is masked out anyway)
    l_safe = tl.where(l_prev > 0.0, l_prev, 1.0)
    O_final = O_acc / l_safe[:, None]
    tl.store(
        O_ptr + rs_flat[:, None] * Q_DIM + qh_flat[:, None] * D + rd[None, :],
        O_final.to(tl.bfloat16),
        mask=qd_mask,
    )


# ---------------------------------------------------------------------------
# Python API
# ---------------------------------------------------------------------------


def prepare_fused_qkv_weight(layer):
    """Concatenate q_proj, k_proj, v_proj weights into (QKV_DIM, hidden_size)."""
    attn = layer.self_attn if hasattr(layer, "self_attn") else layer
    q_w = attn.q_proj.weight.data
    k_w = attn.k_proj.weight.data
    v_w = attn.v_proj.weight.data
    return torch.cat([q_w, k_w, v_w], dim=0).contiguous()


def prepare_signed_sin(sin, head_dim):
    """Pre-apply rotate_half sign pattern to sin (shared across layers).

    signed_sin[m, d] = -sin[m, d] for d < half, +sin[m, d] for d >= half.
    """
    half = head_dim // 2
    sign = torch.ones(head_dim, device=sin.device, dtype=sin.dtype)
    sign[:half] = -1.0
    return (sign.unsqueeze(0) * sin).contiguous()


def prepare_norm_weight_rot(norm_weight, head_dim):
    """Rotated norm weight for cross-term (per layer).

    w_rot[d] = w[d+half] for d < half, w[d-half] for d >= half.
    """
    half = head_dim // 2
    return torch.cat([norm_weight[half:], norm_weight[:half]]).contiguous()


def _run_direct(
    qkv,
    q_norm_w,
    q_norm_w_rot,
    k_norm_w,
    k_norm_w_rot,
    cos,
    signed_sin,
    num_heads,
    num_kv_heads,
    head_dim,
    out,
):
    M = qkv.shape[0]
    Q_DIM = num_heads * head_dim
    K_DIM = num_kv_heads * head_dim
    QKV_DIM = Q_DIM + 2 * K_DIM
    GROUP_SIZE = num_heads // num_kv_heads
    fused_llm_attention_direct_kernel[lambda meta: (triton.cdiv(M, meta["BLOCK_S"]), num_kv_heads)](
        qkv,
        out,
        q_norm_w,
        q_norm_w_rot,
        k_norm_w,
        k_norm_w_rot,
        cos,
        signed_sin,
        M,
        QKV_DIM=QKV_DIM,
        Q_DIM=Q_DIM,
        K_DIM=K_DIM,
        D=head_dim,
        NUM_Q_HEADS=num_heads,
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=GROUP_SIZE,
        BLOCK_D=head_dim,
    )
    return out


def _run_split(
    qkv,
    q_norm_w,
    q_norm_w_rot,
    k_norm_w,
    k_norm_w_rot,
    cos,
    signed_sin,
    num_heads,
    num_kv_heads,
    head_dim,
    out,
):
    M = qkv.shape[0]
    Q_DIM = num_heads * head_dim
    K_DIM = num_kv_heads * head_dim
    QKV_DIM = Q_DIM + 2 * K_DIM
    GROUP_SIZE = num_heads // num_kv_heads

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

    fused_llm_attention_split_kernel[
        lambda meta: (
            triton.cdiv(M, meta["BLOCK_S"]),
            num_kv_heads,
            meta["NUM_SPLITS"],
        )
    ](
        qkv,
        O_partial,
        m_partial,
        l_partial,
        q_norm_w,
        q_norm_w_rot,
        k_norm_w,
        k_norm_w_rot,
        cos,
        signed_sin,
        M,
        QKV_DIM=QKV_DIM,
        Q_DIM=Q_DIM,
        K_DIM=K_DIM,
        D=head_dim,
        NUM_Q_HEADS=num_heads,
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=GROUP_SIZE,
        BLOCK_D=head_dim,
    )

    BLOCK_S_R = 32
    fused_llm_attention_reduce_kernel[(triton.cdiv(M, BLOCK_S_R), num_heads)](
        O_partial,
        m_partial,
        l_partial,
        out,
        M,
        NUM_HEADS=num_heads,
        MAX_SPLITS_CE=MAX_SPLITS,
        D=head_dim,
        Q_DIM=Q_DIM,
        BLOCK_S=BLOCK_S_R,
        BLOCK_D=head_dim,
    )
    return out


def autotune_split_m_threshold(workloads, candidates=THRESHOLD_CANDIDATES, warmup=3, iters=10):
    """Pick the best direct-vs-split dispatch threshold for fixed LLM workloads."""
    global SPLIT_M_THRESHOLD
    if not workloads:
        return SPLIT_M_THRESHOLD

    timings = {}
    for workload in workloads:
        (
            qkv,
            q_norm_w,
            q_norm_w_rot,
            k_norm_w,
            k_norm_w_rot,
            cos,
            signed_sin,
            num_heads,
            num_kv_heads,
            head_dim,
        ) = workload
        out = torch.empty(
            (qkv.shape[0], num_heads * head_dim), device=qkv.device, dtype=torch.bfloat16
        )
        path_times = {}
        for name, fn in (("direct", _run_direct), ("split", _run_split)):
            for _ in range(warmup):
                fn(
                    qkv,
                    q_norm_w,
                    q_norm_w_rot,
                    k_norm_w,
                    k_norm_w_rot,
                    cos,
                    signed_sin,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    out,
                )
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn(
                    qkv,
                    q_norm_w,
                    q_norm_w_rot,
                    k_norm_w,
                    k_norm_w_rot,
                    cos,
                    signed_sin,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    out,
                )
            end.record()
            torch.cuda.synchronize()
            path_times[name] = start.elapsed_time(end) / iters
        timings[qkv.shape[0]] = path_times

    best_threshold = SPLIT_M_THRESHOLD
    best_ms = None
    for threshold in candidates:
        total_ms = 0.0
        for workload in workloads:
            M = workload[0].shape[0]
            total_ms += timings[M]["direct" if M >= threshold else "split"]
        if best_ms is None or total_ms < best_ms:
            best_ms = total_ms
            best_threshold = threshold

    SPLIT_M_THRESHOLD = best_threshold
    return best_threshold


def forward(
    qkv,  # (M, QKV_DIM) bf16
    q_norm_w,  # (D,) bf16 — per-layer
    q_norm_w_rot,  # (D,) bf16 — per-layer
    k_norm_w,  # (D,) bf16 — per-layer
    k_norm_w_rot,  # (D,) bf16 — per-layer
    cos,  # (M, D) bf16 — shared
    signed_sin,  # (M, D) bf16 — shared
    num_heads=32,
    num_kv_heads=8,
    head_dim=128,
    out=None,
):
    """GQA fused LLM attention.

    Dispatches on M:
      - M >= SPLIT_M_THRESHOLD → direct kernel (no partials, no reduce launch)
      - M <  SPLIT_M_THRESHOLD → split kernel + reduce kernel (more parallelism)
    """
    M = qkv.shape[0]
    Q_DIM = num_heads * head_dim

    O_out = (
        out if out is not None else torch.empty((M, Q_DIM), device=qkv.device, dtype=torch.bfloat16)
    )

    if M >= SPLIT_M_THRESHOLD:
        return _run_direct(
            qkv,
            q_norm_w,
            q_norm_w_rot,
            k_norm_w,
            k_norm_w_rot,
            cos,
            signed_sin,
            num_heads,
            num_kv_heads,
            head_dim,
            O_out,
        )
    return _run_split(
        qkv,
        q_norm_w,
        q_norm_w_rot,
        k_norm_w,
        k_norm_w_rot,
        cos,
        signed_sin,
        num_heads,
        num_kv_heads,
        head_dim,
        O_out,
    )
