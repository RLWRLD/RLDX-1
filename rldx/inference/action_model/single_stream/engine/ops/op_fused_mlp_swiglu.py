"""Register ss::fused_mlp_swiglu custom op.

Wraps fused_mlp_swiglu.fused_mlp_swiglu:
  GEMM(X, W_mlp) + bias + SwiGLU in a single Triton kernel.
  X is read once (shared between gate and up), intermediate buffer never materialized.
"""

from __future__ import annotations

import torch


@torch.library.custom_op("ss::fused_mlp_swiglu", mutates_args=())
def fused_mlp_swiglu(
    x: torch.Tensor,  # (M, K) bf16
    weight: torch.Tensor,  # (2*N_HALF, K) bf16
    bias: torch.Tensor,  # (2*N_HALF,) bf16
    split_k: int = 1,  # number of K-dimension splits
) -> torch.Tensor:
    """Fused MLP GEMM + SwiGLU with split-K support.

    Computes: SiLU(X @ W_gate.T + bias_gate) * (X @ W_up.T + bias_up)
    where W = [W_gate; W_up] and bias = [bias_gate; bias_up].

    Args:
        x: (M, K) bf16 — input
        weight: (2*N_HALF, K) bf16 — MLP weight [gate(N_HALF, K); up(N_HALF, K)]
        bias: (2*N_HALF,) bf16 — MLP bias [gate(N_HALF); up(N_HALF)]
        split_k: number of K-dimension splits (1 = no split)

    Returns:
        (M, N_HALF) bf16 — SwiGLU output
    """
    from single_stream.engine.kernels.fused_mlp_swiglu import fused_mlp_swiglu as _impl

    return _impl(x, weight, bias, split_k=split_k)


@fused_mlp_swiglu.register_fake
def _(x, weight, bias, split_k=1):
    M = x.shape[0]
    N_HALF = weight.shape[0] // 2
    return x.new_empty((M, N_HALF))
