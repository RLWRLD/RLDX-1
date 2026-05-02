"""Graph-safe wrapper for the Single-Stream (SS) phase of MSAT.

Manages a list of SingleStreamBlock (or ExpandedSingleStreamBlock) instances
with pre-computed static RoPE position IDs.

When n_physics > 0, builds layout [VL_proj + time_token + SA | P] for
ExpandedSingleStreamBlock. When n_physics == 0, builds standard layout
[VL_proj | time_token | SA].
"""

from __future__ import annotations

from action_model.model.graph_safe_msat import GraphSafeRoPEEmbedder1D
import torch
import torch.nn as nn

from rldx.utils.dist import rank_zero_print as _print


class GraphSafeSingleStreamBlock(nn.Module):
    """Graph-safe SS phase: static RoPE + block iteration.

    Args:
        single_blocks: nn.ModuleList of SingleStreamBlock or ExpandedSingleStreamBlock
        rope_embedder: RoPEEmbedder1D from the original MSAT
        n_vl: number of VL (projected) tokens
        n_sa_pure: number of pure SA tokens (excluding time_token)
        num_temb_tokens: number of time embedding tokens (typically 1)
        n_physics: number of physics tokens (0 = no physics)
        device: target device
    """

    def __init__(
        self,
        single_blocks,
        rope_embedder,
        n_vl,
        n_sa_pure,
        num_temb_tokens=1,
        n_physics=0,
        device=None,
    ):
        super().__init__()
        self._blocks = single_blocks
        self.n_vl = n_vl
        self.n_sa_pure = n_sa_pure
        self.num_temb_tokens = num_temb_tokens
        self.n_physics = n_physics

        # --- Static RoPE position IDs ---
        # Layout: [VL_proj(n_vl) | time_token(num_temb) | SA(n_sa_pure) | P?(n_physics)]
        total_x = n_vl + num_temb_tokens + n_sa_pure
        total = total_x + n_physics if n_physics > 0 else total_x

        ids = torch.zeros(1, total, 2, dtype=torch.long, device=device)

        # VL: axis0=0, axis1=0 (identity rotation for rope_sa_only)
        # Time token: axis1 = 0..num_temb_tokens-1
        tt_start = n_vl
        ids[:, tt_start : tt_start + num_temb_tokens, 1] = torch.arange(
            num_temb_tokens, device=device
        )

        # SA: axis1 = starting from (num_temb_tokens + 1) — matches original
        sa_start = tt_start + num_temb_tokens
        ids[:, sa_start : sa_start + n_sa_pure, 1] = torch.arange(
            num_temb_tokens + 1,
            num_temb_tokens + 1 + n_sa_pure,
            device=device,
        )

        # P: axis0=1 (distinguishes from SA+VL), axis1=sequential
        if n_physics > 0:
            p_start = total_x
            ids[:, p_start:, 0] = 1
            ids[:, p_start:, 1] = torch.arange(n_physics, device=device)

        self.register_buffer("static_ids", ids)
        self.rope = GraphSafeRoPEEmbedder1D(rope_embedder, ids)

        _print(
            f"  [GraphSafeSS] ids={list(ids.shape)}, "
            f"n_vl={n_vl}, n_sa_pure={n_sa_pure}, "
            f"num_temb={num_temb_tokens}, n_physics={n_physics}"
        )

    def forward(self, x, temb, time_token, p=None):
        """Run all SS blocks with static RoPE.

        Args:
            x: (B, n_vl + num_temb + n_sa_pure, hidden_size) — concatenated VL+tt+SA
            temb: (B, temb_dim) — timestep embedding
            time_token: (B, num_temb, hidden_size) — time token for identity modulation
            p: (B, n_physics, sa_dim) or None — physics tokens

        Returns:
            x when p is None, else (x, p)
        """
        pe = self.rope()

        if p is not None:
            for blk in self._blocks:
                x, p = blk(x, temb, pe=pe, time_token=time_token, p_tokens=p)
            return x, p
        else:
            for blk in self._blocks:
                x = blk(x, temb, pe=pe, time_token=time_token)
            return x
