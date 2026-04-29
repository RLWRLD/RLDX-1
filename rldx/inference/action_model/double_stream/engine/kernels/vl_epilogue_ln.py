"""
Fused VL epilogue + LayerNorm kernel.

Replaces K22 (VL pointwise: residual + proj_bias + w3_bias + w3_out)
and K03 (VL LayerNorm) at layer boundaries.

Outputs:
  new_vl = residual + proj_bias + w3_bias + w3_out  (for future residual)
  ln_vl  = LayerNorm(new_vl)                         (for next layer's QKV GEMM)

By fusing K22 + K03, we eliminate 1 GMEM read of the VL tensor (128x4096 = 1MB)
at each layer boundary.

1 kernel launch replaces 2 (K22 + K03).
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # BLOCK_D=4096: single-pass for DIM=4096 (all in registers)
        triton.Config({"BLOCK_D": 4096}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_D": 4096}, num_stages=2, num_warps=16),
        triton.Config({"BLOCK_D": 4096}, num_stages=2, num_warps=32),
        # BLOCK_D=2048: 2-tile multi-pass
        triton.Config({"BLOCK_D": 2048}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_D": 2048}, num_stages=2, num_warps=16),
        # BLOCK_D=1024
        triton.Config({"BLOCK_D": 1024}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_D": 1024}, num_stages=2, num_warps=8),
        # Smaller fallback
        triton.Config({"BLOCK_D": 512}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_D": 512}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_D": 256}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_D": 256}, num_stages=2, num_warps=4),
    ],
    key=["M", "DIM"],
)
@triton.jit
def _vl_epilogue_ln_kernel(
    # Inputs
    vl_res_ptr,  # (M, DIM) bf16 — VL residual
    proj_bias_ptr,  # (DIM,) bf16 — VL proj bias
    w3_bias_ptr,  # (DIM,) bf16 — VL w3 bias
    w3_out_ptr,  # (M, DIM) bf16 — VL w3 mm output
    # Outputs
    new_vl_ptr,  # (M, DIM) bf16 — updated VL hidden state
    ln_vl_ptr,  # (M, DIM) bf16 — LayerNorm output
    # Dims
    M: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Fused VL: new_vl = res + proj_bias + w3_bias + w3_out; ln_vl = LN(new_vl).

    Grid: (M,) -- one CTA per row.

    Single-pass (BLOCK_D >= DIM): all values in registers, 1 read + 2 writes.
    Multi-pass (BLOCK_D < DIM): pass1 = mean + store new_vl,
                                 pass2 = variance (read new_vl from L2),
                                 pass3 = normalize + store ln_vl.
    """
    pid = tl.program_id(0)
    eps: tl.constexpr = 1e-6
    row = pid
    off = row * DIM

    if BLOCK_D >= DIM:
        # === Single-pass: everything in registers ===
        rd = tl.arange(0, BLOCK_D)
        mask = rd < DIM

        # CHECKME: bf16 loads + sequential bf16 addition (eager: bf16 tensor ops)
        x = tl.load(vl_res_ptr + off + rd, mask=mask, other=0.0).to(tl.bfloat16)
        pb = tl.load(proj_bias_ptr + rd, mask=mask, other=0.0).to(tl.bfloat16)
        wb = tl.load(w3_bias_ptr + rd, mask=mask, other=0.0).to(tl.bfloat16)
        w3 = tl.load(w3_out_ptr + off + rd, mask=mask, other=0.0).to(tl.bfloat16)

        x = x + pb  # CHECKME: bf16 sequential addition (matches eager rounding)
        x = x + wb
        x = x + w3

        tl.store(new_vl_ptr + off + rd, x, mask=mask)  # CHECKME: bf16 store

        x_ln = x.to(tl.float32)  # LN in fp32 (matches eager)
        mean = tl.sum(tl.where(mask, x_ln, 0.0)) / DIM
        diff = x_ln - mean
        var = tl.sum(tl.where(mask, diff * diff, 0.0)) / DIM
        inv_std = 1.0 / tl.sqrt(var + eps)
        ln = (diff * inv_std).to(tl.bfloat16)
        tl.store(ln_vl_ptr + off + rd, ln, mask=mask)
    else:
        # === Multi-pass ===
        # Pass 1: compute x, store new_vl, accumulate mean
        acc_sum = 0.0
        for d_start in range(0, DIM, BLOCK_D):
            rd = d_start + tl.arange(0, BLOCK_D)
            mask = rd < DIM
            # CHECKME: bf16 loads + sequential bf16 addition (eager: bf16 tensor ops)
            x = tl.load(vl_res_ptr + off + rd, mask=mask, other=0.0).to(tl.bfloat16)
            pb = tl.load(proj_bias_ptr + rd, mask=mask, other=0.0).to(tl.bfloat16)
            wb = tl.load(w3_bias_ptr + rd, mask=mask, other=0.0).to(tl.bfloat16)
            w3 = tl.load(w3_out_ptr + off + rd, mask=mask, other=0.0).to(tl.bfloat16)
            x = x + pb  # CHECKME: bf16 sequential addition (matches eager rounding)
            x = x + wb
            x = x + w3
            tl.store(new_vl_ptr + off + rd, x, mask=mask)  # CHECKME: bf16 store
            acc_sum += tl.sum(tl.where(mask, x.to(tl.float32), 0.0))
        mean = acc_sum / DIM

        # Pass 2: variance (read new_vl from L2 cache)
        acc_var = 0.0
        for d_start in range(0, DIM, BLOCK_D):
            rd = d_start + tl.arange(0, BLOCK_D)
            mask = rd < DIM
            x = tl.load(new_vl_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
            diff_val = tl.where(mask, x - mean, 0.0)
            acc_var += tl.sum(diff_val * diff_val)
        inv_std = 1.0 / tl.sqrt(acc_var / DIM + eps)

        # Pass 3: normalize + store ln_vl
        for d_start in range(0, DIM, BLOCK_D):
            rd = d_start + tl.arange(0, BLOCK_D)
            mask = rd < DIM
            x = tl.load(new_vl_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
            ln = ((x - mean) * inv_std).to(tl.bfloat16)
            tl.store(ln_vl_ptr + off + rd, ln, mask=mask)


def fused_vl_epilogue_ln(
    vl_residual, proj_bias, w3_bias, w3_out, M=None, DIM=None, new_vl_out=None, ln_vl_out=None
):
    """
    Fused VL epilogue + LayerNorm.

    new_vl = vl_residual + proj_bias + w3_bias + w3_out
    ln_vl  = LayerNorm(new_vl)   (no affine, eps=1e-6)

    Args:
        vl_residual: (1, M, DIM)  bf16 -- VL tokens (arg1_1)
        proj_bias:   (DIM,)       bf16 -- vl_proj.bias (arg14_1)
        w3_bias:     (DIM,)       bf16 -- vl_mlp.w3.bias (arg22_1)
        w3_out:      (M, DIM)     bf16 -- VL w3 matmul output (buf60)

    Returns:
        new_vl: (1, M, DIM) bf16 -- updated VL hidden state
        ln_vl:  (1, M, DIM) bf16 -- LayerNorm(new_vl) for next layer's QKV GEMM
    """
    if M is None:
        M = vl_residual.shape[1] if vl_residual.dim() == 3 else vl_residual.shape[0]
    if DIM is None:
        DIM = vl_residual.shape[-1]
    vl_flat = vl_residual.view(M, DIM)
    w3_flat = w3_out.view(M, DIM) if w3_out.dim() != 2 else w3_out

    new_vl = (
        new_vl_out
        if new_vl_out is not None
        else torch.empty((1, M, DIM), dtype=torch.bfloat16, device=vl_residual.device)
    )
    ln_vl = (
        ln_vl_out
        if ln_vl_out is not None
        else torch.empty((1, M, DIM), dtype=torch.bfloat16, device=vl_residual.device)
    )

    grid = (M,)
    _vl_epilogue_ln_kernel[grid](
        vl_flat,
        proj_bias,
        w3_bias,
        w3_flat,
        new_vl,
        ln_vl,
        M=M,
        DIM=DIM,
    )

    return new_vl, ln_vl
