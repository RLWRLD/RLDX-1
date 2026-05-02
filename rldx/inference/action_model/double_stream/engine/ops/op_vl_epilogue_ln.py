"""Register ds::vl_epilogue_ln custom op.

Wraps vl_epilogue_ln.fused_vl_epilogue_ln:
  Fused VL epilogue (residual + biases + w3_out) + LayerNorm.
  Returns both new_vl and ln_vl for cross-layer optimization.
"""

from __future__ import annotations

import torch


@torch.library.custom_op("ds::vl_epilogue_ln", mutates_args=())
def vl_epilogue_ln(
    vl_res: torch.Tensor,  # (1, M, DIM) bf16
    proj_bias: torch.Tensor,  # (DIM,) bf16
    w3_bias: torch.Tensor,  # (DIM,) bf16
    w3_out: torch.Tensor,  # (M, DIM) bf16
    m: int,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused VL epilogue + LayerNorm.

    new_vl = vl_res + proj_bias + w3_bias + w3_out
    ln_vl  = LayerNorm(new_vl)

    Returns:
        (new_vl, ln_vl): both (1, M, DIM) bf16
    """
    from double_stream.engine.kernels.vl_epilogue_ln import fused_vl_epilogue_ln

    return fused_vl_epilogue_ln(
        vl_res,
        proj_bias,
        w3_bias,
        w3_out,
        M=m,
        DIM=dim,
    )


@vl_epilogue_ln.register_fake
def _(vl_res, proj_bias, w3_bias, w3_out, m, dim):
    return (
        vl_res.new_empty((1, m, dim)),
        vl_res.new_empty((1, m, dim)),
    )
