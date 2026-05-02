"""Graph-safe wrapper for MSAT (v1.4 action head).

Pre-determines data-dependent variables (position IDs, conditional branches)
in __init__, then uses them as static values in forward.  Replaces the
original MSAT._forward_inner() for CUDA Graph / compile / TRT / CustomOp.

Does NOT bake tensor values (RoPE cos/sin etc.) — only makes the *inputs*
to expensive operations static.  Actual value pre-computation is an engine concern.

Architecture:
  GraphSafeMSAT (orchestrator)
    ├── GraphSafeDoubleStreamBlock  (DS phase with static RoPE)
    └── GraphSafeSingleStreamBlock  (SS phase with static RoPE)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rldx.utils.dist import rank_zero_print as _print


# RoPE Embedder (shared by DS/SS block wrappers)


class GraphSafeRoPEEmbedder1D(nn.Module):
    """RoPEEmbedder1D with pre-computed static PE tensor.

    Pre-computes the complex64 PE at construction time (before any bf16
    conversion can corrupt the imaginary/sin component), then caches it
    as a non-persistent buffer so .to(dtype) won't touch it.
    """

    def __init__(self, rope_embedder, static_ids):
        super().__init__()
        # Use the SAME freqs_cis buffers as the original rope_embedder.
        # These buffers may have been converted to bf16 (losing imaginary part)
        # when the model was loaded with torch_dtype=bfloat16.  We must use
        # the identical (potentially corrupted) values to match vanilla exactly.
        device = static_ids.device

        freqs_list = []
        for i in range(rope_embedder.n_axes):
            freqs_cis = getattr(rope_embedder, f"freqs_cis_{i}").to(device)
            pos_ids = static_ids[..., i]
            freqs_list.append(freqs_cis[pos_ids])
        pe = torch.cat(freqs_list, dim=-1)  # (B, N, D//2)

        # persistent=False: .to(dtype) won't convert this buffer
        self.register_buffer("static_pe", pe, persistent=False)

    def forward(self):
        """Return pre-computed complex64 PE tensor."""
        return self.static_pe


# MSAT (orchestrator)


class GraphSafeMSAT(nn.Module):
    """Graph-safe MSAT wrapper — orchestrates DS and SS block phases.

    Delegates RoPE computation and block iteration to:
      - GraphSafeDoubleStreamBlock (DS phase)
      - GraphSafeSingleStreamBlock (SS phase)

    Handles:
      - Timestep encoding + time token construction
      - Time token prepend / strip between DS and SS phases
      - VL projection to SA dim
      - Output projection (action + optional physics)

    Args:
        msat: original MSAT module
        n_vl: number of VL (encoder) tokens
        n_sa_pure: number of pure SA tokens (excluding time_token)
        device: target device
        n_physics: number of physics tokens (0 = v1.4-v, >0 = v1.4-all)
    """

    def __init__(self, msat, n_vl, n_sa_pure, device, n_physics=0):
        super().__init__()
        self._msat = msat
        self.n_vl = n_vl
        self.n_sa_pure = n_sa_pure
        self.n_physics = n_physics
        self.num_temb_tokens = msat.num_temb_tokens  # 1

        n_sa = n_sa_pure + self.num_temb_tokens  # e.g. 18 = 17 + 1

        # --- Block-level wrappers (manage their own RoPE) ---
        from double_stream.model.graph_safe_double_stream import GraphSafeDoubleStreamBlock
        from single_stream.model.graph_safe_single_stream import GraphSafeSingleStreamBlock

        self.gs_ds = GraphSafeDoubleStreamBlock(
            msat.double_blocks,
            msat.rope_embedder,
            n_vl=n_vl,
            n_sa=n_sa,
            n_physics=n_physics,
            device=device,
        )
        self.gs_ss = GraphSafeSingleStreamBlock(
            msat.single_blocks,
            msat.rope_embedder,
            n_vl=n_vl,
            n_sa_pure=n_sa_pure,
            num_temb_tokens=self.num_temb_tokens,
            n_physics=n_physics,
            device=device,
        )

        _print(
            f"  [GraphSafeMSAT] n_vl={n_vl}, n_sa_pure={n_sa_pure}, "
            f"n_physics={n_physics}, num_temb={self.num_temb_tokens}"
        )

    def forward(self, hidden_states, encoder_hidden_states, timestep, physics_embs=None):
        """Graph-safe MSAT forward.

        Args:
            hidden_states: (B, n_sa_pure, sa_dim) — SA tokens (state + action)
            encoder_hidden_states: (B, n_vl, vl_dim) — VL tokens
            timestep: (B,) — timestep values
            physics_embs: (B, n_physics, sa_dim) or None — physics tokens

        Returns:
            action output tensor when physics_embs is None,
            {"action": ..., "physics": ...} dict when physics_embs is not None.
        """
        msat = self._msat
        sa = hidden_states
        vl = encoder_hidden_states
        p = physics_embs

        # 1. Timestep encoding
        temb = msat.timestep_encoder(timestep)

        # 2. Time token (has_time_token=True, no branching)
        time_token = msat.time_token_proj(temb).unsqueeze(1)
        time_token = time_token.repeat(1, self.num_temb_tokens, 1)
        sa = torch.cat([time_token, sa], dim=1)

        # 3. DS phase
        if p is not None:
            sa, vl, p = self.gs_ds(sa, vl, temb, p=p)
        else:
            sa, vl = self.gs_ds(sa, vl, temb)

        # 4. Strip time token (static offset)
        time_token = sa[:, : self.num_temb_tokens]
        sa = sa[:, self.num_temb_tokens :]

        # 5. Project VL → SS concat
        vl_projected = msat.vl_proj_to_sa(vl)
        x = torch.cat([vl_projected, time_token, sa], dim=1)

        # 6. SS phase
        if p is not None:
            x, p = self.gs_ss(x, temb, time_token, p=p)
        else:
            x = self.gs_ss(x, temb, time_token)

        # 7. Extract SA from SS output
        sa = x[:, -self.n_sa_pure :]

        # 8. Output projection (action)
        shift, scale = msat.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        sa = msat.norm_out(sa) * (1 + scale[:, None]) + shift[:, None]
        action_out = msat.proj_out_2(sa)

        # 9. Physics output projection (if applicable)
        if p is not None:
            p_shift, p_scale = msat.proj_out_physics_1(F.silu(temb)).chunk(2, dim=1)
            p = msat.norm_out_physics(p) * (1 + p_scale[:, None]) + p_shift[:, None]
            physics_out = msat.proj_out_physics_2(p)
            return {"action": action_out, "physics": physics_out}

        return action_out

    def __getattr__(self, name):
        """Delegate attribute access to the original MSAT model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._msat, name)
