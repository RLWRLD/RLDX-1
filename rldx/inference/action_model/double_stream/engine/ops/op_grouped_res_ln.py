"""Register ds::grouped_res_ln custom op.

Wraps grouped_res_ln.fused_grouped_res_ln:
  Grouped SA + VL residual + proj_bias + LayerNorm.
"""

from __future__ import annotations

import torch


@torch.library.custom_op("ds::grouped_res_ln", mutates_args=())
def grouped_res_ln(
    sa_res: torch.Tensor,  # (1, M_sa, SA_DIM) bf16
    sa_bias: torch.Tensor,  # (SA_DIM,) bf16
    sa_proj: torch.Tensor,  # (M_sa, SA_DIM) bf16
    vl_res: torch.Tensor,  # (1, M_vl, VL_DIM) bf16
    vl_bias: torch.Tensor,  # (VL_DIM,) bf16
    vl_proj: torch.Tensor,  # (M_vl, VL_DIM) bf16
    m_sa: int,
    m_vl: int,
    sa_dim: int,
    vl_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grouped SA + VL residual + proj_bias + LayerNorm.

    For each stream: output = LayerNorm(residual + proj_output + proj_bias)

    Returns:
        (sa_out, vl_out): (1, M_sa, SA_DIM) and (1, M_vl, VL_DIM) bf16
    """
    from double_stream.engine.kernels.grouped_res_ln import fused_grouped_res_ln

    return fused_grouped_res_ln(
        sa_res,
        sa_bias,
        sa_proj,
        vl_res,
        vl_bias,
        vl_proj,
        M_sa=m_sa,
        M_vl=m_vl,
        SA_DIM=sa_dim,
        VL_DIM=vl_dim,
    )


@grouped_res_ln.register_fake
def _(sa_res, sa_bias, sa_proj, vl_res, vl_bias, vl_proj, m_sa, m_vl, sa_dim, vl_dim):
    return (
        sa_res.new_empty((1, m_sa, sa_dim)),
        vl_res.new_empty((1, m_vl, vl_dim)),
    )
