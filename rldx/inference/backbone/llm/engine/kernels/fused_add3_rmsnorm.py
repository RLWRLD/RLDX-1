"""Fused 3-way residual add + RMSNorm Triton kernel (with DeepStack).

    new_hidden = hidden + residual + deepstack
    normed     = RMSNorm(new_hidden, weight)

Single GMEM pass: reads hidden/residual/deepstack once, writes new_hidden and normed.
Saves one full GMEM read compared to separate add + RMSNorm.

RMSNorm precision matches eager:
  1. variance = mean(x^2) in fp32
  2. inv_rms = rsqrt(variance + eps) in fp32
  3. (x * inv_rms).to(bf16) * weight.to(bf16) — cast BEFORE weight

Grid: (M,) — one CTA per row (token).
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": bd, "NUM_STAGES": ns}, num_stages=ns, num_warps=nw)
        for bd in [256, 512, 1024, 2048, 4096]
        for ns in [1, 2, 3, 4]
        for nw in [4, 8, 16, 32]
    ],
    key=["D"],
)
@triton.jit
def _fused_add3_rmsnorm_kernel(
    hidden_ptr,
    residual_ptr,
    deepstack_ptr,
    weight_ptr,
    new_hidden_ptr,
    normed_ptr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    row = tl.program_id(0)
    eps: tl.constexpr = 1e-6
    off = row * D

    if BLOCK_D >= D:
        rd = tl.arange(0, BLOCK_D)
        mask = rd < D

        h = tl.load(hidden_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(residual_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
        ds = tl.load(deepstack_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + rd, mask=mask, other=0.0)

        new_h = h + r + ds
        tl.store(new_hidden_ptr + off + rd, new_h.to(tl.bfloat16), mask=mask)

        variance = tl.sum(tl.where(mask, new_h * new_h, 0.0)) / D
        inv_rms = tl.rsqrt(variance + eps)
        normed = (new_h * inv_rms).to(tl.bfloat16) * w.to(tl.bfloat16)
        tl.store(normed_ptr + off + rd, normed, mask=mask)
    else:
        acc_sq = 0.0
        for d_start in tl.range(0, D, BLOCK_D, num_stages=NUM_STAGES):
            rd = d_start + tl.arange(0, BLOCK_D)
            mask = rd < D
            h = tl.load(hidden_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
            r = tl.load(residual_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
            ds = tl.load(deepstack_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
            new_h = h + r + ds
            tl.store(new_hidden_ptr + off + rd, new_h.to(tl.bfloat16), mask=mask)
            acc_sq += tl.sum(tl.where(mask, new_h * new_h, 0.0))

        inv_rms = tl.rsqrt(acc_sq / D + eps)

        for d_start in tl.range(0, D, BLOCK_D, num_stages=NUM_STAGES):
            rd = d_start + tl.arange(0, BLOCK_D)
            mask = rd < D
            x = tl.load(new_hidden_ptr + off + rd, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(weight_ptr + rd, mask=mask, other=0.0)
            normed = (x * inv_rms).to(tl.bfloat16) * w.to(tl.bfloat16)
            tl.store(normed_ptr + off + rd, normed, mask=mask)


def forward(hidden, residual, deepstack, weight):
    """Fused 3-way residual add + RMSNorm (with DeepStack).

    Args:
        hidden:    (*, D) bf16 — layer output (e.g. mlp_out)
        residual:  (*, D) bf16 — skip connection
        deepstack: (*, D) bf16 — DeepStack visual features
        weight:    (D,) bf16 — RMSNorm weight

    Returns:
        new_hidden: (*, D) bf16 — hidden + residual + deepstack
        normed:     (*, D) bf16 — RMSNorm(new_hidden)
    """
    shape = hidden.shape
    D = shape[-1]
    M = hidden.numel() // D

    h_flat = hidden.reshape(M, D)
    r_flat = residual.reshape(M, D)
    ds_flat = deepstack.reshape(M, D)

    new_hidden = torch.empty_like(h_flat)
    normed = torch.empty_like(h_flat)

    _fused_add3_rmsnorm_kernel[(M,)](
        h_flat,
        r_flat,
        ds_flat,
        weight,
        new_hidden,
        normed,
        D=D,
    )
    return new_hidden.view(shape), normed.view(shape)
