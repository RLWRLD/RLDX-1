"""Graph-safe wrapper for the Double-Stream (DS) phase of MSAT.

Manages a list of DoubleStreamBlock (or ExpandedDoubleStreamBlock) instances
with pre-computed static RoPE position IDs.

When n_physics > 0, builds 3-way RoPE layout [VL | SA | P] for
ExpandedDoubleStreamBlock. When n_physics == 0, builds standard 2-way
layout [VL | SA] for DoubleStreamBlock.
"""

from __future__ import annotations

from action_model.model.graph_safe_msat import GraphSafeRoPEEmbedder1D
import torch
import torch.nn as nn

from rldx.utils.dist import rank_zero_print as _print


class GraphSafeDoubleStreamBlock(nn.Module):
    """Graph-safe DS phase: static RoPE + block iteration.

    Args:
        double_blocks: nn.ModuleList of DoubleStreamBlock or ExpandedDoubleStreamBlock
        rope_embedder: RoPEEmbedder1D from the original MSAT
        n_vl: number of VL (encoder) tokens
        n_sa: number of SA tokens **including** time_token (n_sa_pure + num_temb_tokens)
        n_physics: number of physics tokens (0 = no physics)
        device: target device
    """

    def __init__(self, double_blocks, rope_embedder, n_vl, n_sa, n_physics=0, device=None):
        super().__init__()
        self._blocks = double_blocks
        self.n_vl = n_vl
        self.n_sa = n_sa
        self.n_physics = n_physics

        # --- Static RoPE position IDs ---
        if n_physics > 0:
            # 3-way layout: [VL(n_vl) | SA(n_sa) | P(n_physics)]
            total = n_vl + n_sa + n_physics
            ids = torch.zeros(1, total, 2, dtype=torch.long, device=device)

            # SA positions on axis1 (time_token at 0, then 1..n_sa-1)
            sa_start = n_vl
            ids[:, sa_start : sa_start + n_sa, 1] = torch.arange(n_sa, device=device)

            # P positions: axis0=1 (distinguishes from SA), axis1=sequential
            p_start = n_vl + n_sa
            ids[:, p_start:, 0] = 1
            ids[:, p_start:, 1] = torch.arange(n_physics, device=device)
        else:
            # 2-way layout: [VL(n_vl) | SA(n_sa)]
            total = n_vl + n_sa
            ids = torch.zeros(1, total, 2, dtype=torch.long, device=device)

            # SA positions on axis1
            ids[:, n_vl:, 1] = torch.arange(n_sa, device=device)

        self.register_buffer("static_ids", ids)
        self.rope = GraphSafeRoPEEmbedder1D(rope_embedder, ids)

        _print(
            f"  [GraphSafeDS] ids={list(ids.shape)}, "
            f"n_vl={n_vl}, n_sa={n_sa}, n_physics={n_physics}"
        )

    def forward(self, sa, vl, temb, p=None):
        """Run all DS blocks with static RoPE.

        Args:
            sa: (B, n_sa, sa_dim) — SA tokens with time_token already prepended
            vl: (B, n_vl, vl_dim) — VL tokens
            temb: (B, temb_dim) — timestep embedding
            p:  (B, n_physics, sa_dim) or None — physics tokens

        Returns:
            (sa, vl) when p is None, else (sa, vl, p)
        """
        pe = self.rope()

        if p is not None:
            for blk in self._blocks:
                sa, vl, p = blk(
                    sa,
                    vl,
                    temb,
                    pe=pe,
                    has_time_token=True,
                    p_tokens=p,
                )
            return sa, vl, p
        else:
            for blk in self._blocks:
                sa, vl = blk(sa, vl, temb, pe=pe, has_time_token=True)
            return sa, vl
