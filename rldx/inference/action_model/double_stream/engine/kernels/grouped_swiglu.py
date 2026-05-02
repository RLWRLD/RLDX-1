"""
Grouped SA + VL SwiGLU kernel.

Replaces K19 (SA SwiGLU) + K21 (VL SwiGLU) with 1 kernel launch.

SwiGLU formula:
  silu_part = w12_out[:, :N_half] + w12_bias[:N_half]
  gate_part = w12_out[:, N_half:] + w12_bias[N_half:]
  output = SiLU(silu_part) * gate_part

SA and VL may have different N_half (SA: 4096, VL: 10922).

Uses 1D linearized grid to avoid wasted tiles:
  SA_total = cdiv(M_sa, BLOCK_M) * cdiv(N_half_sa, BLOCK_N)
  VL_total = cdiv(M_vl, BLOCK_M) * cdiv(N_half_vl, BLOCK_N)
  Grid: (SA_total + VL_total,)
Each CTA decomposes its linear pid into (group, pid_m, pid_n).

2 kernel launches → 1.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # --- BLOCK_M=16 ---
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_stages=5, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_stages=4, num_warps=16),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_stages=3, num_warps=16),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 2048}, num_stages=2, num_warps=16),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 4096}, num_stages=2, num_warps=16),
        # --- BLOCK_M=32 ---
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_stages=5, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_stages=5, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_stages=4, num_warps=16),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 1024}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 1024}, num_stages=3, num_warps=16),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 2048}, num_stages=2, num_warps=16),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 4096}, num_stages=2, num_warps=16),
        # --- BLOCK_M=64 ---
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_stages=5, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_stages=5, num_warps=16),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 512}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 512}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 512}, num_stages=4, num_warps=16),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 1024}, num_stages=2, num_warps=16),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 1024}, num_stages=3, num_warps=16),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 2048}, num_stages=2, num_warps=16),
        # --- BLOCK_M=128 ---
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=5, num_warps=16),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_stages=4, num_warps=16),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 512}, num_stages=2, num_warps=16),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 512}, num_stages=3, num_warps=16),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 1024}, num_stages=2, num_warps=16),
        # --- BLOCK_M=256 ---
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128}, num_stages=3, num_warps=16),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256}, num_stages=2, num_warps=16),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256}, num_stages=3, num_warps=16),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 512}, num_stages=2, num_warps=16),
    ],
    key=["M_sa", "M_vl", "N_half_sa", "N_half_vl"],
)
@triton.jit
def _grouped_swiglu_kernel(
    # SA pointers
    sa_w12_out_ptr,  # (M_sa, 2*N_half_sa) bf16
    sa_bias_ptr,  # (2*N_half_sa,) bf16
    sa_output_ptr,  # (M_sa, N_half_sa) bf16
    # VL pointers
    vl_w12_out_ptr,  # (M_vl, 2*N_half_vl) bf16
    vl_bias_ptr,  # (2*N_half_vl,) bf16
    vl_output_ptr,  # (M_vl, N_half_vl) bf16
    # Dimensions
    M_sa: tl.constexpr,  # 18
    M_vl: tl.constexpr,  # 256
    N_half_sa: tl.constexpr,  # 4096
    N_half_vl: tl.constexpr,  # 10922
    # Tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Grouped SA + VL SwiGLU with different N_half per group.

    1D linearized grid: (SA_TOTAL_TILES + VL_TOTAL_TILES,)
    pid < SA_TOTAL_TILES  →  SA group
    pid >= SA_TOTAL_TILES →  VL group
    """
    SA_N_TILES: tl.constexpr = (N_half_sa + BLOCK_N - 1) // BLOCK_N
    SA_M_TILES: tl.constexpr = (M_sa + BLOCK_M - 1) // BLOCK_M
    SA_TOTAL_TILES: tl.constexpr = SA_M_TILES * SA_N_TILES
    VL_N_TILES: tl.constexpr = (N_half_vl + BLOCK_N - 1) // BLOCK_N

    pid = tl.program_id(0)

    if pid < SA_TOTAL_TILES:
        pid_m = pid // SA_N_TILES
        pid_n = pid % SA_N_TILES

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = rm < M_sa
        mask_n = rn < N_half_sa
        mn_mask = mask_m[:, None] & mask_n[None, :]

        STRIDE_W12_SA: tl.constexpr = 2 * N_half_sa

        silu_raw = tl.load(
            sa_w12_out_ptr + rm[:, None] * STRIDE_W12_SA + rn[None, :],
            mask=mn_mask,
            other=0.0,
        )
        silu_bias = tl.load(sa_bias_ptr + rn, mask=mask_n, other=0.0)
        silu_val = silu_raw.to(tl.float32) + silu_bias.to(tl.float32)[None, :]
        silu_act = silu_val * tl.sigmoid(silu_val)

        gate_raw = tl.load(
            sa_w12_out_ptr + rm[:, None] * STRIDE_W12_SA + (N_half_sa + rn)[None, :],
            mask=mn_mask,
            other=0.0,
        )
        gate_bias = tl.load(sa_bias_ptr + N_half_sa + rn, mask=mask_n, other=0.0)
        gate_val = gate_raw.to(tl.float32) + gate_bias.to(tl.float32)[None, :]

        output = silu_act * gate_val
        tl.store(
            sa_output_ptr + rm[:, None] * N_half_sa + rn[None, :],
            output.to(tl.bfloat16),
            mask=mn_mask,
        )
    else:
        linear = pid - SA_TOTAL_TILES
        pid_m = linear // VL_N_TILES
        pid_n = linear % VL_N_TILES

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = rm < M_vl
        mask_n = rn < N_half_vl
        mn_mask = mask_m[:, None] & mask_n[None, :]

        STRIDE_W12_VL: tl.constexpr = 2 * N_half_vl

        silu_raw = tl.load(
            vl_w12_out_ptr + rm[:, None] * STRIDE_W12_VL + rn[None, :],
            mask=mn_mask,
            other=0.0,
        )
        silu_bias = tl.load(vl_bias_ptr + rn, mask=mask_n, other=0.0)
        silu_val = silu_raw.to(tl.float32) + silu_bias.to(tl.float32)[None, :]
        silu_act = silu_val * tl.sigmoid(silu_val)

        gate_raw = tl.load(
            vl_w12_out_ptr + rm[:, None] * STRIDE_W12_VL + (N_half_vl + rn)[None, :],
            mask=mn_mask,
            other=0.0,
        )
        gate_bias = tl.load(vl_bias_ptr + N_half_vl + rn, mask=mask_n, other=0.0)
        gate_val = gate_raw.to(tl.float32) + gate_bias.to(tl.float32)[None, :]

        output = silu_act * gate_val
        tl.store(
            vl_output_ptr + rm[:, None] * N_half_vl + rn[None, :],
            output.to(tl.bfloat16),
            mask=mn_mask,
        )


def fused_grouped_swiglu(
    sa_w12_out, sa_bias, vl_w12_out, vl_bias, M_sa=None, M_vl=None, N_half_sa=None, N_half_vl=None
):
    """
    Grouped SA + VL SwiGLU.

    Args:
        sa_w12_out: (M_sa, 2*N_half_sa) bf16 — SA w12 matmul output
        sa_bias:    (2*N_half_sa,) bf16 — SA w12 bias
        vl_w12_out: (M_vl, 2*N_half_vl) bf16 — VL w12 matmul output
        vl_bias:    (2*N_half_vl,) bf16 — VL w12 bias

    Returns:
        sa_out: (M_sa, N_half_sa) bf16
        vl_out: (M_vl, N_half_vl) bf16
    """
    if M_sa is None:
        M_sa = sa_w12_out.shape[0]
    if N_half_sa is None:
        N_half_sa = sa_w12_out.shape[1] // 2
    if M_vl is None:
        M_vl = vl_w12_out.shape[0]
    if N_half_vl is None:
        N_half_vl = vl_w12_out.shape[1] // 2
    sa_out = torch.empty((M_sa, N_half_sa), dtype=torch.bfloat16, device=sa_w12_out.device)
    vl_out = torch.empty((M_vl, N_half_vl), dtype=torch.bfloat16, device=vl_w12_out.device)

    grid = lambda meta: (
        triton.cdiv(M_sa, meta["BLOCK_M"]) * triton.cdiv(N_half_sa, meta["BLOCK_N"])
        + triton.cdiv(M_vl, meta["BLOCK_M"]) * triton.cdiv(N_half_vl, meta["BLOCK_N"]),
    )

    _grouped_swiglu_kernel[grid](
        sa_w12_out,
        sa_bias,
        sa_out,
        vl_w12_out,
        vl_bias,
        vl_out,
        M_sa=M_sa,
        M_vl=M_vl,
        N_half_sa=N_half_sa,
        N_half_vl=N_half_vl,
    )

    return sa_out, vl_out
