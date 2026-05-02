"""CustomOpExpandedMSAT: 3-way DS + SS custom ops (v1.4-all, with physics).

Uses FullExpandedCustomOpDSChain + FullExpandedCustomOpSSChain with baked RoPE.
temb/time_token are received from ActionModel chain (pre-baked at init).
For v1.4-v (no physics), see custom_msat_chain.py.
"""

from __future__ import annotations

from double_stream.engine.custom_expanded_double_stream_chain import FullExpandedCustomOpDSChain
from single_stream.engine.custom_expanded_single_stream_chain import FullExpandedCustomOpSSChain
import torch
import torch.nn as nn
import torch.nn.functional as F

from .custom_msat_chain import rope_cos_sin_from_embedder


class CustomOpExpandedMSAT(nn.Module):
    """MSAT with 3-way Triton custom ops (v1.4-all, physics stream).

    temb and time_token are passed in (pre-baked by ActionModel chain).
    """

    def __init__(self, gs_msat, n_sa_pure, n_vl, n_physics, device, dtype=torch.bfloat16):
        super().__init__()

        msat = gs_msat._msat
        self.vl_proj_to_sa = msat.vl_proj_to_sa
        self.norm_out = msat.norm_out
        self.proj_out_1 = msat.proj_out_1
        self.proj_out_2 = msat.proj_out_2
        self.norm_out_physics = msat.norm_out_physics
        self.proj_out_physics_1 = msat.proj_out_physics_1
        self.proj_out_physics_2 = msat.proj_out_physics_2
        self.num_temb_tokens = msat.num_temb_tokens
        self.n_sa_pure = n_sa_pure

        n_sa = n_sa_pure + self.num_temb_tokens
        rope = msat.rope_embedder

        # DS: SA RoPE (axis0=0) + P RoPE (axis0=1)
        ds_sa_cos, ds_sa_sin = rope_cos_sin_from_embedder(
            rope, torch.arange(n_sa, device=device), device, axis0_value=0
        )
        ds_p_cos, ds_p_sin = rope_cos_sin_from_embedder(
            rope, torch.arange(n_physics, device=device), device, axis0_value=1
        )
        self.ds_chain = FullExpandedCustomOpDSChain(
            list(msat.double_blocks),
            ds_sa_cos,
            ds_sa_sin,
            ds_p_cos,
            ds_p_sin,
            n_sa,
            n_vl,
            n_physics,
        ).eval()

        # SS: SA RoPE (axis0=0) + P RoPE (axis0=1)
        ss_sa_positions = torch.cat(
            [
                torch.tensor([0], device=device),
                torch.arange(
                    self.num_temb_tokens + 1, self.num_temb_tokens + 1 + n_sa_pure, device=device
                ),
            ]
        )
        ss_sa_cos, ss_sa_sin = rope_cos_sin_from_embedder(
            rope, ss_sa_positions, device, axis0_value=0
        )
        ss_p_cos, ss_p_sin = rope_cos_sin_from_embedder(
            rope, torch.arange(n_physics, device=device), device, axis0_value=1
        )
        self.ss_chain = FullExpandedCustomOpSSChain(
            list(msat.single_blocks),
            ss_sa_cos,
            ss_sa_sin,
            ss_p_cos,
            ss_p_sin,
            n_vl + n_sa,
            n_sa,
            n_physics,
        ).eval()

    def forward(
        self, hidden_states, encoder_hidden_states, temb, time_token, physics_embs, **kwargs
    ):
        sa = hidden_states
        vl = encoder_hidden_states
        p = physics_embs

        sa = torch.cat([time_token, sa], dim=1)

        sa, vl, p = self.ds_chain(sa, vl, p)

        time_token = sa[:, : self.num_temb_tokens]
        sa = sa[:, self.num_temb_tokens :]
        vl_projected = self.vl_proj_to_sa(vl)

        x = torch.cat([vl_projected, time_token, sa], dim=1)
        x, p = self.ss_chain(x, p)
        sa = x[:, -self.n_sa_pure :]

        # Action output
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        sa = self.norm_out(sa) * (1 + scale[:, None]) + shift[:, None]
        action_out = self.proj_out_2(sa)

        # Physics output
        p_shift, p_scale = self.proj_out_physics_1(F.silu(temb)).chunk(2, dim=1)
        p = self.norm_out_physics(p) * (1 + p_scale[:, None]) + p_shift[:, None]
        physics_out = self.proj_out_physics_2(p)

        return {"action": action_out, "physics": physics_out}
