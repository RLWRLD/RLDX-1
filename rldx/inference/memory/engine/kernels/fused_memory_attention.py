"""
Fused kernel: RoPE + Block-Causal Attention for TransformerMemory.

Single forward() call fusing:
  1. Q/K RoPE application (bf16 × bf16, matching eager)
  2. Block-causal SDPA with online softmax

Dtype (matching eager exactly):
  - QKV from GEMM:   bf16
  - RoPE cos/sin:    bf16 (eager: fp32 compute → cast to input dtype bf16)
  - Q * cos + rot(Q) * signed_sin: bf16 × bf16 → bf16
  - QK^T:            bf16 × bf16, fp32 accumulator (tensor core)
  - Softmax:         fp32
  - P @ V:           bf16 × bf16, fp32 accumulator
  - Output:          bf16

No QK RMSNorm (unlike VLM LLM decoder).
No GQA (num_heads == num_kv_heads == 16).

Block-causal: position i attends to j iff i // block_attn_size >= j // block_attn_size

Grid: (cdiv(M, BLOCK_S), NUM_HEADS)
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton Kernel
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_S": bs, "BLOCK_P": bp}, num_stages=ns, num_warps=nw)
        for bs in [16, 32, 64]
        for bp in [16, 32, 64]
        for ns in [2, 3]
        for nw in [4, 8]
    ],
    key=["M"],
)
@triton.jit
def fused_memory_attention_kernel(
    # --- pointers ---
    QKV_ptr,  # (M, QKV_DIM) bf16 — fused QKV GEMM output
    O_ptr,  # (M, Q_DIM) bf16 — attention output
    cos_ptr,  # (M, D) bf16 — shared RoPE cos
    signed_sin_ptr,  # (M, D) bf16 — shared signed sin (sign folded in)
    # --- runtime dims ---
    M,
    # --- constexpr model dims ---
    QKV_DIM: tl.constexpr,
    Q_DIM: tl.constexpr,  # NUM_HEADS * D
    K_DIM: tl.constexpr,  # NUM_HEADS * D (same, no GQA)
    D: tl.constexpr,  # head_dim (256)
    NUM_HEADS: tl.constexpr,
    BLOCK_ATN: tl.constexpr,  # block_attn_size (16)
    # --- autotune block sizes ---
    BLOCK_S: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,  # == D
):
    pid_s = tl.program_id(0)
    head = tl.program_id(1)

    s_start = pid_s * BLOCK_S
    rs = s_start + tl.arange(0, BLOCK_S)
    rd = tl.arange(0, BLOCK_D)

    s_mask = rs < M
    d_mask = rd < D
    sd_mask = s_mask[:, None] & d_mask[None, :]

    # Rotation index for rotate_half
    HALF: tl.constexpr = D // 2
    rd_rot = tl.where(rd < HALF, rd + HALF, rd - HALF)

    D_float: tl.constexpr = D * 1.0
    scale = 1.0 / tl.sqrt(D_float)

    # Column offsets into fused QKV buffer
    q_col = head * D
    k_col = Q_DIM + head * D
    v_col = Q_DIM + K_DIM + head * D

    # ================================================================
    # Phase 1: Q RoPE (all bf16, matching eager apply_rotary_pos_emb)
    # ================================================================
    q_self = tl.load(
        QKV_ptr + rs[:, None] * QKV_DIM + (q_col + rd)[None, :],
        mask=sd_mask,
        other=0.0,
    )  # bf16

    q_cross = tl.load(
        QKV_ptr + rs[:, None] * QKV_DIM + (q_col + rd_rot)[None, :],
        mask=sd_mask,
        other=0.0,
    )  # bf16

    cos_q = tl.load(cos_ptr + rs[:, None] * D + rd[None, :], mask=sd_mask, other=0.0)  # bf16
    ssin_q = tl.load(
        signed_sin_ptr + rs[:, None] * D + rd[None, :], mask=sd_mask, other=0.0
    )  # bf16

    # q * cos + rotate_half(q) * sin — bf16 (matches eager)
    Q_tile = q_self * cos_q + q_cross * ssin_q

    # ================================================================
    # Phase 2: Online-softmax block-causal attention
    # ================================================================
    m_prev = tl.full([BLOCK_S], float("-inf"), dtype=tl.float32)
    l_prev = tl.zeros([BLOCK_S], dtype=tl.float32)
    O_acc = tl.zeros([BLOCK_S, BLOCK_D], dtype=tl.float32)

    for p_start in range(0, M, BLOCK_P):
        rp = p_start + tl.arange(0, BLOCK_P)
        p_mask = rp < M
        pd_mask = p_mask[:, None] & d_mask[None, :]

        # --- K RoPE (bf16) ---
        k_self = tl.load(
            QKV_ptr + rp[:, None] * QKV_DIM + (k_col + rd)[None, :],
            mask=pd_mask,
            other=0.0,
        )
        k_cross = tl.load(
            QKV_ptr + rp[:, None] * QKV_DIM + (k_col + rd_rot)[None, :],
            mask=pd_mask,
            other=0.0,
        )

        cos_k = tl.load(cos_ptr + rp[:, None] * D + rd[None, :], mask=pd_mask, other=0.0)
        ssin_k = tl.load(signed_sin_ptr + rp[:, None] * D + rd[None, :], mask=pd_mask, other=0.0)

        K_tile = k_self * cos_k + k_cross * ssin_k  # bf16

        # --- Attention scores: bf16 × bf16 → fp32 ---
        S = tl.dot(Q_tile, tl.trans(K_tile)).to(tl.float32) * scale

        # --- Block-causal mask ---
        block_causal = (rs[:, None] // BLOCK_ATN) >= (rp[None, :] // BLOCK_ATN)
        valid = s_mask[:, None] & p_mask[None, :]
        S = tl.where(block_causal & valid, S, float("-inf"))

        # --- Online softmax (fp32) ---
        m_cur = tl.max(S, axis=1)
        m_new = tl.maximum(m_prev, m_cur)
        alpha = tl.exp(m_prev - m_new)
        P = tl.exp(S - m_new[:, None])
        l_new = alpha * l_prev + tl.sum(P, axis=1)

        # --- V tile + accumulate ---
        V_tile = tl.load(
            QKV_ptr + rp[:, None] * QKV_DIM + (v_col + rd)[None, :],
            mask=pd_mask,
            other=0.0,
        )  # bf16

        # bf16 × bf16, fp32 accumulation (matches Flash Attention 2)
        O_acc = O_acc * alpha[:, None] + tl.dot(P.to(tl.bfloat16), V_tile)

        m_prev = m_new
        l_prev = l_new

    # ================================================================
    # Phase 3: Normalize and store
    # ================================================================
    O_final = O_acc / l_prev[:, None]

    tl.store(
        O_ptr + rs[:, None] * Q_DIM + (head * D + rd)[None, :],
        O_final.to(tl.bfloat16),
        mask=sd_mask,
    )


# ---------------------------------------------------------------------------
# Python API
# ---------------------------------------------------------------------------


def prepare_fused_qkv_weight(layer):
    """Concatenate q_proj, k_proj, v_proj weights into (QKV_DIM, hidden_size)."""
    attn = layer.self_attn
    return torch.cat(
        [attn.q_proj.weight.data, attn.k_proj.weight.data, attn.v_proj.weight.data], dim=0
    ).contiguous()


def prepare_signed_sin(sin, head_dim):
    """Pre-apply rotate_half sign pattern to sin.

    signed_sin[m, d] = -sin[m, d] for d < half, +sin[m, d] for d >= half.
    Matches eager: rotate_half produces [-x2, x1].
    """
    half = head_dim // 2
    sign = torch.ones(head_dim, device=sin.device, dtype=sin.dtype)
    sign[:half] = -1.0
    return (sign.unsqueeze(0) * sin).contiguous()


def prepare_rope_buffers(rotary_emb, position_ids, device, dtype=torch.bfloat16):
    """Pre-compute RoPE cos, signed_sin from RotaryEmbedding + position_ids.

    Args:
        rotary_emb: RotaryEmbedding module (from any decoder layer)
        position_ids: (1, M) long — block-wise position IDs
        device: torch device
        dtype: output dtype (bf16)

    Returns:
        cos: (M, D) bf16
        signed_sin: (M, D) bf16
    """
    D = rotary_emb.dim
    dummy_x = torch.empty(1, 1, 1, D, device=device, dtype=dtype)
    cos, sin = rotary_emb(dummy_x, position_ids)  # (1, M, D) dtype
    cos = cos.squeeze(0).contiguous()
    sin = sin.squeeze(0)
    ssin = prepare_signed_sin(sin, D)
    return cos, ssin


def forward(qkv, cos, signed_sin, num_heads, head_dim, block_attn_size):
    """Launch fused_memory_attention (RoPE + Block-Causal SDPA).

    Args:
        qkv: (M, QKV_DIM) bf16 — fused QKV GEMM output
        cos: (M, D) bf16 — pre-computed RoPE cos
        signed_sin: (M, D) bf16 — pre-computed signed sin
        num_heads: int (16)
        head_dim: int (256)
        block_attn_size: int (16)

    Returns:
        (M, Q_DIM) bf16 — attention output (before o_proj)
    """
    M = qkv.shape[0]
    Q_DIM = num_heads * head_dim
    K_DIM = num_heads * head_dim
    QKV_DIM = Q_DIM + 2 * K_DIM

    O_out = torch.empty((M, Q_DIM), device=qkv.device, dtype=torch.bfloat16)

    fused_memory_attention_kernel[lambda meta: (triton.cdiv(M, meta["BLOCK_S"]), num_heads)](
        qkv,
        O_out,
        cos,
        signed_sin,
        M,
        QKV_DIM=QKV_DIM,
        Q_DIM=Q_DIM,
        K_DIM=K_DIM,
        D=head_dim,
        NUM_HEADS=num_heads,
        BLOCK_ATN=block_attn_size,
        BLOCK_D=head_dim,
    )

    return O_out
