"""Register ds::grouped_swiglu custom op.

Wraps grouped_swiglu.fused_grouped_swiglu:
  Grouped SA + VL SwiGLU in one kernel launch.
"""

from __future__ import annotations

import torch


@torch.library.custom_op("ds::grouped_swiglu", mutates_args=())
def grouped_swiglu(
    sa_w12: torch.Tensor,  # (M_sa, 2*N_half_sa) bf16
    sa_bias: torch.Tensor,  # (2*N_half_sa,) bf16
    vl_w12: torch.Tensor,  # (M_vl, 2*N_half_vl) bf16
    vl_bias: torch.Tensor,  # (2*N_half_vl,) bf16
    m_sa: int,
    m_vl: int,
    n_half_sa: int,
    n_half_vl: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grouped SA + VL SwiGLU.

    SwiGLU: SiLU(x[:, :N_half] + bias[:N_half]) * (x[:, N_half:] + bias[N_half:])

    Returns:
        (sa_out, vl_out): (M_sa, N_half_sa) and (M_vl, N_half_vl) bf16
    """
    from double_stream.engine.kernels.grouped_swiglu import fused_grouped_swiglu

    return fused_grouped_swiglu(
        sa_w12,
        sa_bias,
        vl_w12,
        vl_bias,
        M_sa=m_sa,
        M_vl=m_vl,
        N_half_sa=n_half_sa,
        N_half_vl=n_half_vl,
    )


@grouped_swiglu.register_fake
def _(sa_w12, sa_bias, vl_w12, vl_bias, m_sa, m_vl, n_half_sa, n_half_vl):
    return (
        sa_w12.new_empty((m_sa, n_half_sa)),
        vl_w12.new_empty((m_vl, n_half_vl)),
    )
