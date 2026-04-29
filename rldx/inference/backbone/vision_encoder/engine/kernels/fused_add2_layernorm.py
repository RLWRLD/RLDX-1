"""Fused 2-way residual add + LayerNorm Triton kernel for Vision encoder.

    new_hidden = hidden + residual
    normed     = LayerNorm(new_hidden, weight, bias)

Single GMEM pass: reads hidden/residual once, writes new_hidden and normed.
Saves one full GMEM read compared to separate add + LayerNorm.

LayerNorm (with affine):
  1. mean = mean(x) in fp32
  2. var = mean((x - mean)^2) in fp32
  3. normed = (x - mean) / sqrt(var + eps) * weight + bias

Grid: (M,) — one CTA per row (token). D=1152 for Qwen3VL vision.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": bd}, num_stages=ns, num_warps=nw)
        for bd in [256, 512, 1024, 2048, 4096]
        for ns in [1, 2, 3, 4]
        for nw in [4, 8, 16, 32]
    ],
    key=["D"],
)
@triton.jit
def _fused_add2_layernorm_kernel(
    hidden_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    new_hidden_ptr,
    normed_ptr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    eps: tl.constexpr = 1e-6
    off = row * D

    if BLOCK_D >= D:
        # Single-pass: all D elements in registers
        rd = tl.arange(0, BLOCK_D)
        mask = rd < D

        h = tl.load(hidden_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(residual_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + rd, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(bias_ptr + rd, mask=mask, other=0.0).to(tl.float32)

        new_h = h + r
        tl.store(new_hidden_ptr + off + rd, new_h.to(tl.bfloat16), mask=mask)

        # LayerNorm in fp32
        mean = tl.sum(tl.where(mask, new_h, 0.0)) / D
        diff = new_h - mean
        var = tl.sum(tl.where(mask, diff * diff, 0.0)) / D
        inv_std = tl.rsqrt(var + eps)
        normed = (diff * inv_std * w + b).to(tl.bfloat16)
        tl.store(normed_ptr + off + rd, normed, mask=mask)
    else:
        # Multi-pass (2-pass with Welford online mean+variance)

        # Pass 1: load h+r → new_hidden + Welford accumulate (mean, M2)
        welford_mean = 0.0
        welford_m2 = 0.0
        welford_count = 0.0
        for d_start in range(0, D, BLOCK_D):
            rd = d_start + tl.arange(0, BLOCK_D)
            mask = rd < D
            h = tl.load(hidden_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
            r = tl.load(residual_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
            new_h = h + r
            tl.store(new_hidden_ptr + off + rd, new_h.to(tl.bfloat16), mask=mask)
            # Welford online: accumulate mean and M2 for valid elements
            n_valid = tl.sum(mask.to(tl.float32))
            tile_sum = tl.sum(tl.where(mask, new_h, 0.0))
            tile_sq_sum = tl.sum(tl.where(mask, new_h * new_h, 0.0))
            # Parallel Welford merge: combine tile stats with running stats
            new_count = welford_count + n_valid
            delta = tile_sum / tl.maximum(n_valid, 1.0) - welford_mean
            welford_mean = welford_mean + delta * n_valid / tl.maximum(new_count, 1.0)
            welford_m2 = (
                welford_m2
                + tile_sq_sum
                - tile_sum * tile_sum / tl.maximum(n_valid, 1.0)
                + delta * delta * welford_count * n_valid / tl.maximum(new_count, 1.0)
            )
            welford_count = new_count

        mean = welford_mean
        var = welford_m2 / D
        inv_std = tl.rsqrt(var + eps)

        # Pass 2: normalize + store (single GMEM read of new_hidden)
        for d_start in range(0, D, BLOCK_D):
            rd = d_start + tl.arange(0, BLOCK_D)
            mask = rd < D
            x = tl.load(new_hidden_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(weight_ptr + rd, mask=mask, other=0.0).to(tl.float32)
            b = tl.load(bias_ptr + rd, mask=mask, other=0.0).to(tl.float32)
            normed = ((x - mean) * inv_std * w + b).to(tl.bfloat16)
            tl.store(normed_ptr + off + rd, normed, mask=mask)


def forward(hidden, residual, weight, bias):
    """Fused 2-way residual add + LayerNorm.

    Args:
        hidden:   (*, D) bf16 — layer output (e.g. attn_out or mlp_out)
        residual: (*, D) bf16 — skip connection
        weight:   (D,) — LayerNorm weight
        bias:     (D,) — LayerNorm bias

    Returns:
        new_hidden: (*, D) bf16 — hidden + residual
        normed:     (*, D) bf16 — LayerNorm(new_hidden)
    """
    shape = hidden.shape
    D = shape[-1]
    M = hidden.numel() // D

    h_flat = hidden.reshape(M, D)
    r_flat = residual.reshape(M, D)

    new_hidden = torch.empty_like(h_flat)
    normed = torch.empty_like(h_flat)

    _fused_add2_layernorm_kernel[(M,)](
        h_flat,
        r_flat,
        weight,
        bias,
        new_hidden,
        normed,
        D=D,
    )
    return new_hidden.view(shape), normed.view(shape)
