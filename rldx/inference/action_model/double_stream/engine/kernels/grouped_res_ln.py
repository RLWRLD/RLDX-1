"""
Fused SA + VL residual + proj_bias + LayerNorm (grouped pointwise + reduction).

Replaces K14 (SA) + K17 (VL) with 1 kernel launch.
  K14: buf47 = LayerNorm(arg0_1 + buf43 + arg12_1)  — SA (18, 1536)
  K17: buf54 = LayerNorm(arg1_1 + buf50 + arg14_1)  — VL (128, 4096)

LayerNorm is without affine (elementwise_affine=False): (x - mean) / sqrt(var + eps)

2 kernel launches -> 1.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 128}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_D": 256}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_D": 256}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_D": 512}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_D": 512}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_D": 1024}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_D": 1024}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_D": 2048}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_D": 2048}, num_stages=2, num_warps=16),
        triton.Config({"BLOCK_D": 4096}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_D": 4096}, num_stages=2, num_warps=16),
        triton.Config({"BLOCK_D": 4096}, num_stages=2, num_warps=32),
        triton.Config({"BLOCK_D": 128}, num_stages=4, num_warps=2),
        triton.Config({"BLOCK_D": 256}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_D": 512}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_D": 512}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_D": 1024}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_D": 2048}, num_stages=4, num_warps=16),
    ],
    key=["M_SA", "M_VL", "SA_DIM", "VL_DIM"],
)
@triton.jit
def _grouped_res_ln_kernel(
    # SA pointers
    sa_res_ptr,  # (M_SA, SA_DIM) bf16 — residual (arg0_1)
    sa_bias_ptr,  # (SA_DIM,)      bf16 — sa_proj.bias (arg12_1)
    sa_proj_ptr,  # (M_SA, SA_DIM) bf16 — proj output (buf43)
    sa_out_ptr,  # (M_SA, SA_DIM) bf16 — output (buf47)
    # VL pointers
    vl_res_ptr,  # (M_VL, VL_DIM) bf16 — residual (arg1_1)
    vl_bias_ptr,  # (VL_DIM,)      bf16 — vl_proj.bias (arg14_1)
    vl_proj_ptr,  # (M_VL, VL_DIM) bf16 — proj output (buf50)
    vl_out_ptr,  # (M_VL, VL_DIM) bf16 — output (buf54)
    # Dims
    M_SA: tl.constexpr,
    M_VL: tl.constexpr,
    SA_DIM: tl.constexpr,
    VL_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Grouped SA + VL: output = LayerNorm(residual + proj_output + proj_bias).

    Grid: (M_SA + M_VL,)
    pid < M_SA  -> SA row (dim=SA_DIM)
    pid >= M_SA -> VL row (dim=VL_DIM)
    """
    pid = tl.program_id(0)
    eps: tl.constexpr = 1e-6

    if pid < M_SA:
        row = pid
        off = row * SA_DIM

        # Pass 1: mean
        acc_sum = 0.0
        for d_start in range(0, SA_DIM, BLOCK_D):
            rd = d_start + tl.arange(0, BLOCK_D)
            mask = rd < SA_DIM
            res = tl.load(sa_res_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
            proj = tl.load(sa_proj_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
            bias = tl.load(sa_bias_ptr + rd, mask=mask, other=0.0).to(tl.float32)
            x = res + proj + bias
            acc_sum += tl.sum(tl.where(mask, x, 0.0))
        mean = acc_sum / SA_DIM

        # Pass 2: variance
        acc_var = 0.0
        for d_start in range(0, SA_DIM, BLOCK_D):
            rd = d_start + tl.arange(0, BLOCK_D)
            mask = rd < SA_DIM
            res = tl.load(sa_res_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
            proj = tl.load(sa_proj_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
            bias = tl.load(sa_bias_ptr + rd, mask=mask, other=0.0).to(tl.float32)
            x = res + proj + bias
            diff = tl.where(mask, x - mean, 0.0)
            acc_var += tl.sum(diff * diff)
        inv_std = 1.0 / tl.sqrt(acc_var / SA_DIM + eps)

        # Pass 3: normalize + store
        for d_start in range(0, SA_DIM, BLOCK_D):
            rd = d_start + tl.arange(0, BLOCK_D)
            mask = rd < SA_DIM
            res = tl.load(sa_res_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
            proj = tl.load(sa_proj_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
            bias = tl.load(sa_bias_ptr + rd, mask=mask, other=0.0).to(tl.float32)
            x = res + proj + bias
            out = ((x - mean) * inv_std).to(tl.bfloat16)
            tl.store(sa_out_ptr + off + rd, out, mask=mask)
    else:
        row = pid - M_SA
        off = row * VL_DIM

        # Pass 1: mean
        acc_sum = 0.0
        for d_start in range(0, VL_DIM, BLOCK_D):
            rd = d_start + tl.arange(0, BLOCK_D)
            mask = rd < VL_DIM
            res = tl.load(vl_res_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
            proj = tl.load(vl_proj_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
            bias = tl.load(vl_bias_ptr + rd, mask=mask, other=0.0).to(tl.float32)
            x = res + proj + bias
            acc_sum += tl.sum(tl.where(mask, x, 0.0))
        mean = acc_sum / VL_DIM

        # Pass 2: variance
        acc_var = 0.0
        for d_start in range(0, VL_DIM, BLOCK_D):
            rd = d_start + tl.arange(0, BLOCK_D)
            mask = rd < VL_DIM
            res = tl.load(vl_res_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
            proj = tl.load(vl_proj_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
            bias = tl.load(vl_bias_ptr + rd, mask=mask, other=0.0).to(tl.float32)
            x = res + proj + bias
            diff = tl.where(mask, x - mean, 0.0)
            acc_var += tl.sum(diff * diff)
        inv_std = 1.0 / tl.sqrt(acc_var / VL_DIM + eps)

        # Pass 3: normalize + store
        for d_start in range(0, VL_DIM, BLOCK_D):
            rd = d_start + tl.arange(0, BLOCK_D)
            mask = rd < VL_DIM
            res = tl.load(vl_res_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
            proj = tl.load(vl_proj_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
            bias = tl.load(vl_bias_ptr + rd, mask=mask, other=0.0).to(tl.float32)
            x = res + proj + bias
            out = ((x - mean) * inv_std).to(tl.bfloat16)
            tl.store(vl_out_ptr + off + rd, out, mask=mask)


def fused_grouped_res_ln(
    sa_residual,
    sa_proj_bias,
    sa_proj_out,
    vl_residual,
    vl_proj_bias,
    vl_proj_out,
    M_sa=None,
    M_vl=None,
    SA_DIM=None,
    VL_DIM=None,
):
    """
    Fused SA + VL residual + proj_bias + LayerNorm.

    For each stream: output = LayerNorm(residual + proj_output + proj_bias)

    Args:
        sa_residual:  (1, M_sa, SA_DIM)  bf16 — SA tokens (arg0_1)
        sa_proj_bias: (SA_DIM,)           bf16 — sa_proj.bias (arg12_1)
        sa_proj_out:  (M_sa, SA_DIM)      bf16 — SA proj matmul output (buf43)
        vl_residual:  (1, M_vl, VL_DIM)  bf16 — VL tokens (arg1_1)
        vl_proj_bias: (VL_DIM,)           bf16 — vl_proj.bias (arg14_1)
        vl_proj_out:  (M_vl, VL_DIM)      bf16 — VL proj matmul output (buf50)

    Returns:
        sa_out: (1, M_sa, SA_DIM) bf16 — buf47
        vl_out: (1, M_vl, VL_DIM) bf16 — buf54
    """
    if M_sa is None:
        M_sa = sa_residual.shape[1] if sa_residual.dim() == 3 else sa_residual.shape[0]
    if SA_DIM is None:
        SA_DIM = sa_residual.shape[-1]
    if M_vl is None:
        M_vl = vl_residual.shape[1] if vl_residual.dim() == 3 else vl_residual.shape[0]
    if VL_DIM is None:
        VL_DIM = vl_residual.shape[-1]
    empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda

    SA_NUMEL = M_sa * SA_DIM
    VL_NUMEL = M_vl * VL_DIM

    # Flatten residuals to 2D
    sa_res_flat = sa_residual.view(M_sa, SA_DIM)
    vl_res_flat = vl_residual.view(M_vl, VL_DIM)
    # proj outputs are already 2D from proj_matmul
    sa_proj_flat = sa_proj_out.view(M_sa, SA_DIM) if sa_proj_out.dim() != 2 else sa_proj_out
    vl_proj_flat = vl_proj_out.view(M_vl, VL_DIM) if vl_proj_out.dim() != 2 else vl_proj_out

    # Allocate outputs matching inductor layout
    sa_out = empty_strided_cuda((1, M_sa, SA_DIM), (SA_NUMEL, SA_DIM, 1), torch.bfloat16)
    vl_out = empty_strided_cuda((1, M_vl, VL_DIM), (VL_NUMEL, VL_DIM, 1), torch.bfloat16)

    grid = (M_sa + M_vl,)

    _grouped_res_ln_kernel[grid](
        sa_res_flat,
        sa_proj_bias,
        sa_proj_flat,
        sa_out,
        vl_res_flat,
        vl_proj_bias,
        vl_proj_flat,
        vl_out,
        M_SA=M_sa,
        M_VL=M_vl,
        SA_DIM=SA_DIM,
        VL_DIM=VL_DIM,
    )

    return sa_out, vl_out
