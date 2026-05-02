"""Fused MLP GEMM + SwiGLU Triton kernel with split-K support.

Computes: output = SiLU(X @ W_gate.T + bias_gate) * (X @ W_up.T + bias_up)

Where W = [W_gate; W_up] is (2*N_HALF, K) and bias = [bias_gate; bias_up].
X is read once per tile (shared between gate and up accumulations),
and the intermediate (M, 2*N_HALF) buffer is never materialized.

Split-K: the K dimension is partitioned across SPLIT_K thread-block groups.
  - SPLIT_K=1: direct output with fused epilogue (no workspace)
  - SPLIT_K>1: partial sums to workspace, then reduction kernel applies epilogue

bf16 stores, fp32 accumulation.
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# Main kernel: GEMM with paired gate/up accumulation
# ============================================================================


@triton.autotune(
    configs=[
        # M~274, K=1536, N_HALF=6144
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 128}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=4, num_warps=8),
    ],
    key=["M", "K", "N_HALF", "SPLIT_K"],
)
@triton.jit
def _fused_mlp_swiglu_kernel(
    # Output: (M, N_HALF) bf16 — only written when SPLIT_K == 1
    OUT_ptr,
    out_stride_m,
    out_stride_n,
    # Workspace: (SPLIT_K, M, N_HALF) fp32 — only written when SPLIT_K > 1
    GATE_WS_ptr,
    UP_WS_ptr,
    # Input: (M, K) bf16
    X_ptr,
    x_stride_m,
    x_stride_k,
    # Weight: (2*N_HALF, K) bf16 — gate=[0:N_HALF], up=[N_HALF:]
    W_ptr,
    w_stride_n,
    w_stride_k,
    # Bias: (2*N_HALF,) bf16
    BIAS_ptr,
    # Dimensions
    M,
    K: tl.constexpr,
    N_HALF: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused GEMM + SwiGLU (split-K).

    Grid: (ceil(M / BLOCK_M), ceil(N_HALF / BLOCK_N), SPLIT_K)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # K range for this split
    k_per_split = (K + SPLIT_K - 1) // SPLIT_K
    k_start = pid_k * k_per_split
    k_end = min(k_start + k_per_split, K)
    k_iters = (k_end - k_start + BLOCK_K - 1) // BLOCK_K

    # Initial pointers (offset to k_start)
    x_ptrs = X_ptr + offs_m[:, None] * x_stride_m + (k_start + offs_k)[None, :] * x_stride_k
    gate_w_ptrs = W_ptr + offs_n[None, :] * w_stride_n + (k_start + offs_k)[:, None] * w_stride_k
    up_w_ptrs = (
        W_ptr + (offs_n[None, :] + N_HALF) * w_stride_n + (k_start + offs_k)[:, None] * w_stride_k
    )

    # Accumulate gate and up in parallel
    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(k_iters):
        k_mask = (k_start + offs_k) < k_end
        m_mask = offs_m[:, None] < M
        n_mask = offs_n[None, :] < N_HALF

        x_tile = tl.load(x_ptrs, mask=m_mask & k_mask[None, :], other=0.0)
        gate_w = tl.load(gate_w_ptrs, mask=k_mask[:, None] & n_mask, other=0.0)
        up_w = tl.load(up_w_ptrs, mask=k_mask[:, None] & n_mask, other=0.0)

        gate_acc += tl.dot(x_tile, gate_w)
        up_acc += tl.dot(x_tile, up_w)

        x_ptrs += BLOCK_K * x_stride_k
        gate_w_ptrs += BLOCK_K * w_stride_k
        up_w_ptrs += BLOCK_K * w_stride_k
        k_start += BLOCK_K
        offs_k += BLOCK_K

    if SPLIT_K == 1:
        # Direct epilogue: bias + SwiGLU → output (bf16)
        n_mask = offs_n < N_HALF
        gate_bias = tl.load(BIAS_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        up_bias = tl.load(BIAS_ptr + offs_n + N_HALF, mask=n_mask, other=0.0).to(tl.float32)
        gate_acc += gate_bias[None, :]
        up_acc += up_bias[None, :]
        result = (gate_acc * tl.sigmoid(gate_acc)) * up_acc
        out_ptrs = OUT_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n
        out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_HALF)
        tl.store(out_ptrs, result.to(tl.bfloat16), mask=out_mask)
    else:
        # Write fp32 partials to workspace: (SPLIT_K, M, N_HALF) contiguous
        ws_off = pid_k * M * N_HALF + offs_m[:, None] * N_HALF + offs_n[None, :]
        ws_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_HALF)
        tl.store(GATE_WS_ptr + ws_off, gate_acc, mask=ws_mask)
        tl.store(UP_WS_ptr + ws_off, up_acc, mask=ws_mask)


# ============================================================================
# Reduction kernel: sum partials + bias + SwiGLU → output
# ============================================================================


@triton.jit
def _reduce_swiglu_kernel(
    # Output: (M, N_HALF) bf16
    OUT_ptr,
    out_stride_m,
    out_stride_n,
    # Workspace: (SPLIT_K, M, N_HALF) fp32
    GATE_WS_ptr,
    UP_WS_ptr,
    # Bias: (2*N_HALF,) bf16
    BIAS_ptr,
    # Dimensions
    M,
    N_HALF: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Reduce split-K partials + apply bias + SwiGLU.

    Grid: (ceil(M / BLOCK_M), ceil(N_HALF / BLOCK_N))
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_HALF)

    # Sum partials over SPLIT_K
    gate_sum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_sum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(SPLIT_K):
        ws_off = k * M * N_HALF + offs_m[:, None] * N_HALF + offs_n[None, :]
        gate_sum += tl.load(GATE_WS_ptr + ws_off, mask=mask, other=0.0)
        up_sum += tl.load(UP_WS_ptr + ws_off, mask=mask, other=0.0)

    # Epilogue: bias + SwiGLU
    n_mask = offs_n < N_HALF
    gate_bias = tl.load(BIAS_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    up_bias = tl.load(BIAS_ptr + offs_n + N_HALF, mask=n_mask, other=0.0).to(tl.float32)
    gate_sum += gate_bias[None, :]
    up_sum += up_bias[None, :]
    result = (gate_sum * tl.sigmoid(gate_sum)) * up_sum

    # Store bf16
    out_ptrs = OUT_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n
    tl.store(out_ptrs, result.to(tl.bfloat16), mask=mask)


# ============================================================================
# Launcher
# ============================================================================


def fused_mlp_swiglu(x, weight, bias, split_k=1):
    """Launch fused MLP GEMM + SwiGLU kernel.

    Args:
        x: (M, K) bf16 — input (pre-norm output, squeezed)
        weight: (2*N_HALF, K) bf16 — MLP weight [gate; up]
        bias: (2*N_HALF,) bf16 — MLP bias [gate; up]
        split_k: number of K-dimension splits (1 = no split)

    Returns:
        (M, N_HALF) bf16 — SwiGLU output
    """
    M, K = x.shape
    N_HALF = weight.shape[0] // 2
    out = torch.empty((M, N_HALF), device=x.device, dtype=x.dtype)

    if split_k == 1:
        # No workspace needed — epilogue fused in main kernel
        gate_ws = torch.empty(0, device=x.device, dtype=torch.float32)
        up_ws = torch.empty(0, device=x.device, dtype=torch.float32)
    else:
        gate_ws = torch.empty((split_k, M, N_HALF), device=x.device, dtype=torch.float32)
        up_ws = torch.empty((split_k, M, N_HALF), device=x.device, dtype=torch.float32)

    grid_main = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N_HALF, meta["BLOCK_N"]),
        split_k,
    )

    _fused_mlp_swiglu_kernel[grid_main](
        out,
        out.stride(0),
        out.stride(1),
        gate_ws,
        up_ws,
        x,
        x.stride(0),
        x.stride(1),
        weight,
        weight.stride(0),
        weight.stride(1),
        bias,
        M=M,
        K=K,
        N_HALF=N_HALF,
        SPLIT_K=split_k,
    )

    if split_k > 1:
        # Reduction: sum partials + bias + SwiGLU → output
        REDUCE_BM, REDUCE_BN = 32, 128
        grid_reduce = (triton.cdiv(M, REDUCE_BM), triton.cdiv(N_HALF, REDUCE_BN))

        _reduce_swiglu_kernel[grid_reduce](
            out,
            out.stride(0),
            out.stride(1),
            gate_ws,
            up_ws,
            bias,
            M=M,
            N_HALF=N_HALF,
            SPLIT_K=split_k,
            BLOCK_M=REDUCE_BM,
            BLOCK_N=REDUCE_BN,
        )

    return out
