"""
Fused SingleStreamBlock epilogue + LayerNorm kernel for cross-layer optimization.

Two modes controlled by HAS_RESIDUAL:
  HAS_RESIDUAL=True  (seq128, seq256):
    Replaces residual_add kernel (K08/K09) + next layer's K00 (LayerNorm).
    new_hidden = hidden + residual; ln = LN(new_hidden)
    Saves 1 GMEM read (avoids re-reading new_hidden for K00).

  HAS_RESIDUAL=False (seq274, seq146):
    Runs after fused GEMM+residual (K09). Computes LN of the output.
    Next layer skips K00 and uses precomputed LN directly.
    Saves 1 kernel launch per layer boundary.

Returns (new_hidden, ln_out) for use in multi-layer pipelines.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # BLOCK_D=2048: single-pass for DIM=1536 (all in registers)
        triton.Config({"BLOCK_D": 2048}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_D": 2048}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_D": 2048}, num_stages=2, num_warps=16),
        # BLOCK_D=1024: 2-tile fallback
        triton.Config({"BLOCK_D": 1024}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_D": 1024}, num_stages=2, num_warps=8),
        # Smaller blocks
        triton.Config({"BLOCK_D": 512}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_D": 512}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_D": 256}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_D": 256}, num_stages=2, num_warps=4),
    ],
    key=["M", "DIM", "HAS_RESIDUAL"],
)
@triton.jit
def _ss_epilogue_ln_kernel(
    # Inputs
    hidden_ptr,  # (M, DIM) bf16 — linear2 output (Type B) or GEMM+res output (Type A)
    residual_ptr,  # (M, DIM) bf16 — original residual (only read when HAS_RESIDUAL)
    # Outputs
    new_hidden_ptr,  # (M, DIM) bf16 — new hidden state (only written when HAS_RESIDUAL)
    ln_ptr,  # (M, DIM) bf16 — LayerNorm output
    # Dims
    M: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    """
    Grid: (M,) -- one CTA per row.

    HAS_RESIDUAL=True:  x = hidden + residual, store x, LN(x) -> ln
    HAS_RESIDUAL=False: x = hidden (already stored), LN(x) -> ln
    """
    pid = tl.program_id(0)
    eps: tl.constexpr = 1e-6
    row = pid
    off = row * DIM

    if BLOCK_D >= DIM:
        # === Single-pass: everything in registers ===
        rd = tl.arange(0, BLOCK_D)
        mask = rd < DIM

        if HAS_RESIDUAL:
            h = tl.load(hidden_ptr + off + rd, mask=mask, other=0.0).to(tl.bfloat16)
            r = tl.load(residual_ptr + off + rd, mask=mask, other=0.0).to(tl.bfloat16)
            x = h + r  # bf16 addition (matches eager)
            tl.store(new_hidden_ptr + off + rd, x, mask=mask)
            x_ln = x.to(tl.float32)
        else:
            x_ln = tl.load(hidden_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)

        mean = tl.sum(tl.where(mask, x_ln, 0.0)) / DIM
        diff = x_ln - mean
        var = tl.sum(tl.where(mask, diff * diff, 0.0)) / DIM
        inv_std = 1.0 / tl.sqrt(var + eps)
        ln = (diff * inv_std).to(tl.bfloat16)
        tl.store(ln_ptr + off + rd, ln, mask=mask)
    else:
        # === Multi-pass ===
        if HAS_RESIDUAL:
            # Pass 1: compute x = hidden + residual, store new_hidden, accumulate mean
            acc_sum = 0.0
            for d_start in range(0, DIM, BLOCK_D):
                rd = d_start + tl.arange(0, BLOCK_D)
                mask = rd < DIM
                h = tl.load(hidden_ptr + off + rd, mask=mask, other=0.0).to(tl.bfloat16)
                r = tl.load(residual_ptr + off + rd, mask=mask, other=0.0).to(tl.bfloat16)
                x = h + r  # bf16 addition (matches eager)
                tl.store(new_hidden_ptr + off + rd, x, mask=mask)
                acc_sum += tl.sum(tl.where(mask, x.to(tl.float32), 0.0))
            mean = acc_sum / DIM

            # Pass 2: variance (read new_hidden from L2)
            acc_var = 0.0
            for d_start in range(0, DIM, BLOCK_D):
                rd = d_start + tl.arange(0, BLOCK_D)
                mask = rd < DIM
                x = tl.load(new_hidden_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
                diff_val = tl.where(mask, x - mean, 0.0)
                acc_var += tl.sum(diff_val * diff_val)
            inv_std = 1.0 / tl.sqrt(acc_var / DIM + eps)

            # Pass 3: normalize + store ln
            for d_start in range(0, DIM, BLOCK_D):
                rd = d_start + tl.arange(0, BLOCK_D)
                mask = rd < DIM
                x = tl.load(new_hidden_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
                ln = ((x - mean) * inv_std).to(tl.bfloat16)
                tl.store(ln_ptr + off + rd, ln, mask=mask)
        else:
            # Pass 1: mean (read hidden directly)
            acc_sum = 0.0
            for d_start in range(0, DIM, BLOCK_D):
                rd = d_start + tl.arange(0, BLOCK_D)
                mask = rd < DIM
                x = tl.load(hidden_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
                acc_sum += tl.sum(tl.where(mask, x, 0.0))
            mean = acc_sum / DIM

            # Pass 2: variance
            acc_var = 0.0
            for d_start in range(0, DIM, BLOCK_D):
                rd = d_start + tl.arange(0, BLOCK_D)
                mask = rd < DIM
                x = tl.load(hidden_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
                diff_val = tl.where(mask, x - mean, 0.0)
                acc_var += tl.sum(diff_val * diff_val)
            inv_std = 1.0 / tl.sqrt(acc_var / DIM + eps)

            # Pass 3: normalize + store ln
            for d_start in range(0, DIM, BLOCK_D):
                rd = d_start + tl.arange(0, BLOCK_D)
                mask = rd < DIM
                x = tl.load(hidden_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
                ln = ((x - mean) * inv_std).to(tl.bfloat16)
                tl.store(ln_ptr + off + rd, ln, mask=mask)


def fused_ss_epilogue_ln(
    hidden, M=None, DIM=None, residual=None, new_hidden_out=None, ln_out_out=None
):
    """
    Fused SingleStreamBlock epilogue + LayerNorm.

    If residual is provided (Type B: seq128, seq256):
        new_hidden = hidden + residual
        ln = LN(new_hidden)
    If residual is None (Type A: seq274, seq146):
        new_hidden = hidden (pass-through, no copy)
        ln = LN(hidden)

    Args:
        hidden:   (1, M, DIM) or (M, DIM) bf16 — linear2 output or GEMM+res output
        M:        number of tokens (derived from hidden if None)
        DIM:      hidden dimension (derived from hidden if None)
        residual: (1, M, DIM) bf16 — original input for residual add (optional)

    Returns:
        new_hidden: (1, M, DIM) bf16 — updated hidden state
        ln_out:     (1, M, DIM) bf16 — LN(new_hidden) for next layer
    """
    if M is None:
        M = hidden.shape[1] if hidden.dim() == 3 else hidden.shape[0]
    if DIM is None:
        DIM = hidden.shape[-1]
    has_residual = residual is not None

    h_flat = hidden.view(M, DIM) if hidden.dim() != 2 else hidden

    if has_residual:
        r_flat = residual.view(M, DIM) if residual.dim() != 2 else residual
        new_hidden = (
            new_hidden_out
            if new_hidden_out is not None
            else torch.empty((1, M, DIM), dtype=torch.bfloat16, device=hidden.device)
        )
    else:
        r_flat = h_flat  # dummy, not read
        new_hidden = hidden if hidden.dim() == 3 else hidden.view(1, M, DIM)

    ln_out = (
        ln_out_out
        if ln_out_out is not None
        else torch.empty((1, M, DIM), dtype=torch.bfloat16, device=hidden.device)
    )

    grid = (M,)
    _ss_epilogue_ln_kernel[grid](
        h_flat,
        r_flat,
        new_hidden,
        ln_out,
        M=M,
        DIM=DIM,
        HAS_RESIDUAL=has_residual,
    )

    return new_hidden, ln_out
