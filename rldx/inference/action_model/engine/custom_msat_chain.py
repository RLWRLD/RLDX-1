"""CustomOpMSAT: 2-way DS + SS custom ops (v1.4-v, no physics).

Uses FullCustomOpDSChain + FullCustomOpSSChain with baked RoPE cos/sin.
temb/time_token are received from ActionModel chain (pre-baked at init).
For v1.4-all (physics), see custom_expanded_msat_chain.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_cos_sin_from_embedder(rope_embedder, axis1_positions, device, axis0_value=0):
    """Extract RoPE cos/sin from vanilla rope_embedder's buffers.

    Uses the exact same freqs_cis buffers as the vanilla MSAT.  When the
    model is loaded with torch_dtype=bfloat16, the originally-complex64
    buffers are cast to bf16 real tensors (imaginary/sin part is lost).
    The vanilla apply_rotary_emb then operates on these corrupted values.

    To match vanilla exactly, we return:
      - cos = the bf16 buffer values (which are the original cos values)
      - sin = zeros (the imaginary part was discarded by bf16 cast)

    Args:
        rope_embedder: RoPEEmbedder1D from the original MSAT
        axis1_positions: 1D tensor of axis1 position indices
        device: target device
        axis0_value: axis0 position (0 for SA/VL, 1 for physics)

    Returns:
        (cos, sin) each shape (N, head_dim//2)
    """
    positions = torch.as_tensor(axis1_positions, dtype=torch.long, device=device)
    n = positions.shape[0]

    freqs_list = []
    for i in range(rope_embedder.n_axes):
        freqs_cis = getattr(rope_embedder, f"freqs_cis_{i}").to(device)
        if i == 0:
            idx = torch.full((n,), axis0_value, dtype=torch.long, device=device)
        else:
            idx = positions
        freqs_list.append(freqs_cis[idx])

    pe = torch.cat(freqs_list, dim=-1)  # (N, D//2) bf16 real

    if pe.is_complex():
        return pe.real.contiguous(), pe.imag.contiguous()
    else:
        # bf16 cast killed imaginary part — cos only, sin = 0
        return pe.contiguous(), torch.zeros_like(pe)


from double_stream.engine.custom_double_stream_chain import FullCustomOpDSChain  # noqa: E402
from single_stream.engine.custom_single_stream_chain import FullCustomOpSSChain  # noqa: E402


class CustomOpMSAT(nn.Module):
    """MSAT with 2-way Triton custom ops (v1.4-v).

    temb and time_token are passed in (pre-baked by ActionModel chain),
    not computed internally. Only DS/SS chains + output projection here.
    """

    def __init__(self, gs_msat, n_sa_pure, n_vl, device, dtype=torch.bfloat16):
        super().__init__()

        msat = gs_msat._msat
        self.vl_proj_to_sa = msat.vl_proj_to_sa
        self.norm_out = msat.norm_out
        self.proj_out_1 = msat.proj_out_1
        self.proj_out_2 = msat.proj_out_2
        self.num_temb_tokens = msat.num_temb_tokens
        self.n_sa_pure = n_sa_pure

        n_sa = n_sa_pure + self.num_temb_tokens
        rope = msat.rope_embedder

        ds_positions = torch.arange(n_sa, device=device)
        ds_cos, ds_sin = rope_cos_sin_from_embedder(rope, ds_positions, device)
        self.ds_chain = FullCustomOpDSChain(
            list(msat.double_blocks),
            ds_cos,
            ds_sin,
            n_sa,
            n_vl,
        ).eval()

        ss_positions = torch.cat(
            [
                torch.tensor([0], device=device),
                torch.arange(
                    self.num_temb_tokens + 1, self.num_temb_tokens + 1 + n_sa_pure, device=device
                ),
            ]
        )
        ss_cos, ss_sin = rope_cos_sin_from_embedder(rope, ss_positions, device)
        self.ss_chain = FullCustomOpSSChain(
            list(msat.single_blocks),
            ss_cos,
            ss_sin,
            n_vl + n_sa,
            n_sa,
        ).eval()

    def forward(self, hidden_states, encoder_hidden_states, temb, time_token, **kwargs):
        sa = hidden_states
        vl = encoder_hidden_states

        sa = torch.cat([time_token, sa], dim=1)
        sa, vl = self.ds_chain(sa, vl)

        time_token = sa[:, : self.num_temb_tokens]
        sa = sa[:, self.num_temb_tokens :]
        vl_projected = self.vl_proj_to_sa(vl)

        x = torch.cat([vl_projected, time_token, sa], dim=1)
        x = self.ss_chain(x)
        sa = x[:, -self.n_sa_pure :]

        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        sa = self.norm_out(sa) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(sa)
