"""Fused 2-way residual add + RMSNorm for TransformerMemory.

Reuses the generic Triton kernel from VLM LLM engine (parameterized by D).

    new_hidden = hidden + residual
    normed     = RMSNorm(new_hidden, weight)

Dtype (matching eager RMSNorm exactly):
  1. new_hidden = fp32 add, stored as bf16
  2. variance = mean(x^2) in fp32
  3. inv_rms = rsqrt(variance + eps) in fp32
  4. (x * inv_rms).to(bf16) * weight.to(bf16) — cast BEFORE weight multiply
"""

from backbone.llm.engine.kernels.fused_add2_rmsnorm import forward


__all__ = ["forward"]
