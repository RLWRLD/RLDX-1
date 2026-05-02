"""Register ss::fused_epilogue_ln custom op.

Wraps ss_epilogue_ln.fused_ss_epilogue_ln: Residual + LayerNorm.
"""

from __future__ import annotations

import torch


@torch.library.custom_op("ss::fused_epilogue_ln", mutates_args=())
def fused_epilogue_ln(
    hidden: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused residual + LayerNorm (Type B: HAS_RESIDUAL=True).

    Args:
        hidden: (1, M, 1536) bf16 — block output (before residual)
        residual: (1, M, 1536) bf16 — input to the block

    Returns:
        new_hidden: (1, M, 1536) bf16 — hidden + residual
        ln_out: (1, M, 1536) bf16 — LN(new_hidden)
    """
    from single_stream.engine.kernels.ss_epilogue_ln import fused_ss_epilogue_ln

    M = hidden.shape[1]
    return fused_ss_epilogue_ln(hidden, M=M, DIM=1536, residual=residual)


@fused_epilogue_ln.register_fake
def _(hidden, residual):
    return (torch.empty_like(hidden), torch.empty_like(hidden))
