"""MSAT: Multi-Stream Action Transformer (top-level orchestrator).

Submodules:
- msat_attention.py: BasicTransformerBlock, SelfAttentionTransformer
- msat_ops.py: RoPE, TimestepEncoder, SwiGLUFFN, head utilities
- msat_blocks.py: Modulation, SingleStream, DoubleStream, Expanded, TripleStream blocks
"""

from typing import Optional

from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
import torch
from torch import nn
import torch.nn.functional as F

from rldx.model.modules.action_model.attention import (
    BasicTransformerBlock,
    SelfAttentionTransformer,
)
from rldx.model.modules.action_model.blocks import (
    DoubleStreamBlock,
    ExpandedDoubleStreamBlock,
    ExpandedSingleStreamBlock,
    Modulation,
    ModulationOut,
    SingleStreamBlock,
    TripleStreamBlock,
)
from rldx.model.modules.action_model.ops import RoPEEmbedder1D, TimestepEncoder

# Re-export for backward compatibility (external code may import from msat directly)
from rldx.utils.dist import rank_zero_print as _print


__all__ = [
    "BasicTransformerBlock",
    "SelfAttentionTransformer",
    "JointBase",
    "MSAT",
]


class JointBase(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    def _build_double_blocks(
        self,
        depth,
        sa_dim,
        vl_dim,
        num_heads,
        head_dim,
        dropout,
        activation_fn,
        attention_bias,
        norm_eps,
        qk_norm: str = "none",
        use_swiglu: bool = False,
        mlp_ratio: float = 4.0,
        vl_mlp_ratio: Optional[float] = None,
        positional_embeddings: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        temb_type: str = "layerwise_mod",
        remove_bias: bool = False,
        pre_norm: str = "layer_norm",
        post_norm: str = "none",
    ):
        return nn.ModuleList(
            [
                DoubleStreamBlock(
                    sa_dim=sa_dim,
                    vl_dim=vl_dim,
                    num_attention_heads=num_heads,
                    attention_head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    vl_mlp_ratio=vl_mlp_ratio,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_eps=norm_eps,
                    qk_norm=qk_norm,
                    use_swiglu=use_swiglu,
                    positional_embeddings=positional_embeddings,
                    max_seq_length=max_seq_length,
                    temb_type=temb_type,
                    remove_bias=remove_bias,
                    pre_norm=pre_norm,
                    post_norm=post_norm,
                )
                for _ in range(depth)
            ]
        )

    def _build_triple_blocks(
        self,
        depth,
        vl_dim,
        sa_dim,
        p_dim,
        num_heads,
        head_dim,
        dropout,
        activation_fn,
        attention_bias,
        norm_eps,
        qk_norm: str = "none",
        use_swiglu: bool = False,
        mlp_ratio: float = 4.0,
        vl_mlp_ratio: Optional[float] = None,
        sa_mlp_ratio: Optional[float] = None,
        p_mlp_ratio: Optional[float] = None,
        positional_embeddings: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        temb_type: str = "layerwise_mod",
        remove_bias: bool = False,
        pre_norm: str = "layer_norm",
        post_norm: str = "none",
    ):
        return nn.ModuleList(
            [
                TripleStreamBlock(
                    vl_dim=vl_dim,
                    sa_dim=sa_dim,
                    p_dim=p_dim,
                    num_attention_heads=num_heads,
                    attention_head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    vl_mlp_ratio=vl_mlp_ratio,
                    sa_mlp_ratio=sa_mlp_ratio,
                    p_mlp_ratio=p_mlp_ratio,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_eps=norm_eps,
                    qk_norm=qk_norm,
                    use_swiglu=use_swiglu,
                    positional_embeddings=positional_embeddings,
                    max_seq_length=max_seq_length,
                    temb_type=temb_type,
                    remove_bias=remove_bias,
                    pre_norm=pre_norm,
                    post_norm=post_norm,
                )
                for _ in range(depth)
            ]
        )

    def _build_single_blocks(
        self,
        depth,
        hidden_size,
        num_heads,
        head_dim,
        dropout,
        activation_fn,
        attention_bias,
        norm_eps,
        qk_norm: str = "none",
        use_swiglu: bool = False,
        mlp_ratio: float = 4.0,
        positional_embeddings: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        temb_type: str = "layerwise_mod",
        remove_bias: bool = False,
        pre_norm: str = "layer_norm",
        post_norm: str = "none",
    ):
        return nn.ModuleList(
            [
                SingleStreamBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_heads,
                    attention_head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_eps=norm_eps,
                    qk_norm=qk_norm,
                    use_swiglu=use_swiglu,
                    positional_embeddings=positional_embeddings,
                    max_seq_length=max_seq_length,
                    temb_type=temb_type,
                    remove_bias=remove_bias,
                    pre_norm=pre_norm,
                    post_norm=post_norm,
                )
                for _ in range(depth)
            ]
        )

    def _build_expanded_double_blocks(
        self,
        depth,
        sa_dim,
        vl_dim,
        p_dim,
        num_heads,
        head_dim,
        dropout,
        activation_fn,
        attention_bias,
        norm_eps,
        qk_norm="none",
        use_swiglu=False,
        mlp_ratio=4.0,
        vl_mlp_ratio=None,
        p_mlp_ratio=None,
        positional_embeddings=None,
        max_seq_length=None,
        temb_type="layerwise_mod",
        remove_bias=False,
        pre_norm="layer_norm",
        post_norm="none",
    ):
        return nn.ModuleList(
            [
                ExpandedDoubleStreamBlock(
                    sa_dim=sa_dim,
                    vl_dim=vl_dim,
                    p_dim=p_dim,
                    num_attention_heads=num_heads,
                    attention_head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    vl_mlp_ratio=vl_mlp_ratio,
                    p_mlp_ratio=p_mlp_ratio,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_eps=norm_eps,
                    qk_norm=qk_norm,
                    use_swiglu=use_swiglu,
                    positional_embeddings=positional_embeddings,
                    max_seq_length=max_seq_length,
                    temb_type=temb_type,
                    remove_bias=remove_bias,
                    pre_norm=pre_norm,
                    post_norm=post_norm,
                )
                for _ in range(depth)
            ]
        )

    def _build_expanded_single_blocks(
        self,
        depth,
        hidden_size,
        p_dim,
        num_heads,
        head_dim,
        dropout,
        activation_fn,
        attention_bias,
        norm_eps,
        qk_norm="none",
        use_swiglu=False,
        mlp_ratio=4.0,
        positional_embeddings=None,
        max_seq_length=None,
        temb_type="layerwise_mod",
        remove_bias=False,
        pre_norm="layer_norm",
        post_norm="none",
    ):
        return nn.ModuleList(
            [
                ExpandedSingleStreamBlock(
                    hidden_size=hidden_size,
                    p_dim=p_dim,
                    num_attention_heads=num_heads,
                    attention_head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_eps=norm_eps,
                    qk_norm=qk_norm,
                    use_swiglu=use_swiglu,
                    positional_embeddings=positional_embeddings,
                    max_seq_length=max_seq_length,
                    temb_type=temb_type,
                    remove_bias=remove_bias,
                    pre_norm=pre_norm,
                    post_norm=post_norm,
                )
                for _ in range(depth)
            ]
        )

    def _forward_inner(
        self,
        sa_embs,
        vl_embs,
        timesteps,
        return_all_hidden_states=False,
        encoder_attention_mask=None,
        physics_embs=None,
        physics_attention_mask=None,
    ):
        """
        Forward pass for MSAT.
            - sa_embs: (B, N_sa, sa_dim) - concatenated state+action
            - vl_embs: (B, N_vl, vl_dim) - VL tokens (may include cognition tokens if use_cog_tokens=True)
            - physics_embs: (B, N_p, sa_dim) - physics signal tokens (only when use_physics=True)
            - physics_attention_mask: (B,) - per-sample physics mask (1=visible, 0=masked)
        """
        temb = self.timestep_encoder(timesteps)

        # Create Time Token
        time_token = None
        has_time_token = False
        if self.temb_type == "input_token":
            # 1. Projection: (B, D) -> (B, 1, D)
            t_emb = self.time_token_proj(temb).unsqueeze(1)

            # 2. Replication: (B, 1, D) -> (B, N, D)
            time_token = t_emb.repeat(1, self.num_temb_tokens, 1)
            has_time_token = True

        # Track VL length for Single Stream Block RoPE calculation
        N_vl_for_single = None

        # Generate Shared Modulations
        shared_modulations = None
        shared_single_modulation = None
        if self.temb_type == "shared_mod":
            sa_mod1_raw, sa_mod2_raw = self.shared_sa_mod(temb)
            vl_mod1_raw, vl_mod2_raw = self.shared_vl_mod(temb)
            shared_modulations = {
                "sa_mod1_raw": sa_mod1_raw,
                "sa_mod2_raw": sa_mod2_raw,
                "vl_mod1_raw": vl_mod1_raw,
                "vl_mod2_raw": vl_mod2_raw,
            }
            shared_single_mod_raw, _ = self.shared_single_mod(temb)
            if hasattr(self, "shared_single_mod_proj"):
                shared_single_modulation = ModulationOut(
                    shift=self.shared_single_mod_proj(shared_single_mod_raw.shift),
                    scale=self.shared_single_mod_proj(shared_single_mod_raw.scale),
                    gate=self.shared_single_mod_proj(shared_single_mod_raw.gate),
                )
            else:
                shared_single_modulation = shared_single_mod_raw

        sa, vl = sa_embs.contiguous(), vl_embs.contiguous()

        # ══════════════════════════════════════════════════════════════════════
        # Physics-enabled path: ExpandedDouble [VL | SA | P] -> ExpandedSingle [VL+SA | P]
        # ══════════════════════════════════════════════════════════════════════
        if self.use_physics and physics_embs is not None:
            return self._forward_physics(
                sa,
                vl,
                physics_embs,
                temb,
                time_token,
                has_time_token,
                return_all_hidden_states,
                shared_modulations,
                shared_single_modulation,
                encoder_attention_mask,
                physics_attention_mask,
            )

        # ══════════════════════════════════════════════════════════════════════
        # Standard path: DoubleStreamBlocks [VL | SA] -> SingleStreamBlocks [VL_proj | SA]
        # ══════════════════════════════════════════════════════════════════════

        # Prepend time_token to SA: [time_token | S | A]
        if has_time_token:
            sa = torch.cat([time_token, sa], dim=1)

        all_hidden = [sa]

        pe = None
        if self.use_rope:
            B, N_vl = vl.shape[0], vl.shape[1]
            N_sa_total = sa.shape[1]
            device = sa.device

            total_len = N_vl + N_sa_total
            ids = torch.zeros(B, total_len, 2, dtype=torch.long, device=device)
            sa_stream_start_idx = N_vl

            if self.positional_embeddings == "rope_sa_only":
                # Position ID assignment: (time_token) | S | A
                current_idx = sa_stream_start_idx
                if has_time_token:
                    # Time token: axis 1 = 0..num_temb_tokens-1
                    ids[:, current_idx : current_idx + self.num_temb_tokens, 1] = (
                        torch.arange(self.num_temb_tokens, device=device).unsqueeze(0).expand(B, -1)
                    )
                    current_idx += self.num_temb_tokens
                # SA tokens (S | A): axis 1 = starting from (num_temb_tokens)
                sa_len = N_sa_total - (self.num_temb_tokens if has_time_token else 0)
                start_pos = self.num_temb_tokens if has_time_token else 0
                ids[:, current_idx:, 1] = (
                    torch.arange(start_pos, start_pos + sa_len, device=device)
                    .unsqueeze(0)
                    .expand(B, -1)
                )
            elif self.positional_embeddings == "rope_vl_sa":
                ids[:, :N_vl, 0] = torch.arange(N_vl, device=device).unsqueeze(0).expand(B, -1)
                # Position ID assignment: (time_token) | S | A
                current_idx = sa_stream_start_idx
                if has_time_token:
                    # Time token: axis 1 = 0..num_temb_tokens-1
                    ids[:, current_idx : current_idx + self.num_temb_tokens, 1] = (
                        torch.arange(self.num_temb_tokens, device=device).unsqueeze(0).expand(B, -1)
                    )
                    current_idx += self.num_temb_tokens
                # SA tokens (S | A): axis 1 = starting from (num_temb_tokens)
                sa_len = N_sa_total - (self.num_temb_tokens if has_time_token else 0)
                start_pos = (self.num_temb_tokens if has_time_token else 0) + 1
                ids[:, current_idx:, 1] = (
                    torch.arange(start_pos, start_pos + sa_len, device=device)
                    .unsqueeze(0)
                    .expand(B, -1)
                )

            pe = self.rope_embedder(ids)

        # Track block index
        block_idx = 0
        for blk in self.double_blocks:
            sa, vl = blk(
                sa,
                vl,
                temb,
                pe=pe,
                shared_modulations=shared_modulations,
                has_time_token=has_time_token,
                block_idx=block_idx,
                encoder_attention_mask=encoder_attention_mask,
            )
            all_hidden.append(sa)
            block_idx += 1

        # Separate time_token before single stream block
        if has_time_token:
            time_token = sa[:, : self.num_temb_tokens, :]
            sa = sa[:, self.num_temb_tokens :, :]

        if len(self.single_blocks) > 0:
            vl_projected = self.vl_proj_to_sa(vl)
            N_vl_for_single = vl.shape[
                1
            ]  # Track VL length for Single Stream Block RoPE calculation

            # Re-concat with updated time_token: VL | (time_token) | S | A
            if has_time_token:
                x = torch.cat([vl_projected, time_token, sa], dim=1)
            else:
                x = torch.cat([vl_projected, sa], dim=1)

        # Single Stream Blocks
        if len(self.single_blocks) > 0:
            pe_single = None
            if self.use_rope:
                B_single = x.shape[0]
                N_total = x.shape[1]
                device_single = x.device

                N_action_pure = sa.shape[1]
                action_start_idx_in_x = N_total - N_action_pure

                # 2D RoPE
                ids_single = torch.zeros(
                    B_single, N_total, 2, dtype=torch.long, device=device_single
                )

                if self.positional_embeddings == "rope_sa_only":
                    current_idx = N_vl_for_single
                    if has_time_token:
                        # Time token: axis 1 = 0..num_temb_tokens-1
                        ids_single[:, current_idx : current_idx + self.num_temb_tokens, 1] = (
                            torch.arange(self.num_temb_tokens, device=device_single)
                            .unsqueeze(0)
                            .expand(B_single, -1)
                        )
                        current_idx += self.num_temb_tokens
                    # Action positions: axis 1 = sequence position starting from (num_temb_tokens)
                    start_pos = (self.num_temb_tokens if has_time_token else 0) + 1
                    ids_single[:, action_start_idx_in_x:, 1] = (
                        torch.arange(start_pos, start_pos + N_action_pure, device=device_single)
                        .unsqueeze(0)
                        .expand(B_single, -1)
                    )
                elif self.positional_embeddings == "rope_vl_sa":
                    # VL positions: axis 0 = sequence position
                    ids_single[:, :N_vl_for_single, 0] = (
                        torch.arange(N_vl_for_single, device=device_single)
                        .unsqueeze(0)
                        .expand(B_single, -1)
                    )
                    # Context tokens: (time_token) | S | A
                    current_idx = N_vl_for_single
                    if has_time_token:
                        # Time token: axis 1 = 0..num_temb_tokens-1
                        ids_single[:, current_idx : current_idx + self.num_temb_tokens, 1] = (
                            torch.arange(self.num_temb_tokens, device=device_single)
                            .unsqueeze(0)
                            .expand(B_single, -1)
                        )
                        current_idx += self.num_temb_tokens
                    # Action positions: axis 1 = sequence position starting from (num_temb_tokens)
                    start_pos = (self.num_temb_tokens if has_time_token else 0) + 1
                    ids_single[:, action_start_idx_in_x:, 1] = (
                        torch.arange(start_pos, start_pos + N_action_pure, device=device_single)
                        .unsqueeze(0)
                        .expand(B_single, -1)
                    )

                pe_single = self.rope_embedder(ids_single)

            # Build single-stream attention mask from encoder_attention_mask
            single_attn_mask = None
            if encoder_attention_mask is not None:
                B_mask = x.shape[0]
                N_x = x.shape[1]
                N_vl_mask = N_vl_for_single
                N_rest = N_x - N_vl_mask
                rest_mask = torch.ones(
                    B_mask, N_rest, device=x.device, dtype=encoder_attention_mask.dtype
                )
                kv_mask = torch.cat([encoder_attention_mask, rest_mask], dim=1)  # [B, N_x]
                single_attn_mask = kv_mask[:, None, None, :]  # [B, 1, 1, N_x]
                single_attn_mask = torch.where(
                    single_attn_mask == 0,
                    torch.tensor(float("-inf"), device=x.device, dtype=x.dtype),
                    torch.tensor(0.0, device=x.device, dtype=x.dtype),
                )

            # block_idx already tracks the number of DoubleStreamBlocks processed
            for blk in self.single_blocks:
                x = blk(
                    x,
                    temb,
                    pe=pe_single if pe_single is not None else pe,
                    shared_modulation=shared_single_modulation,
                    time_token=time_token if has_time_token else None,
                    block_idx=block_idx,
                    attn_mask=single_attn_mask,
                )
                block_idx += 1

            # Extract Action Part
            N_action_pure = sa.shape[1]
            sa = x[:, -N_action_pure:, :]

        out = self._output_projection(sa, temb)

        if return_all_hidden_states:
            return out, all_hidden
        return out

    def _forward_physics(
        self,
        sa,
        vl,
        p_embs,
        temb,
        time_token,
        has_time_token,
        return_all_hidden_states,
        shared_modulations,
        shared_single_modulation,
        encoder_attention_mask,
        physics_attention_mask=None,
    ):
        """
        Physics-enabled forward:
          Lower: ExpandedDoubleStreamBlocks [VL | SA | P] (3-way via p_tokens kwarg)
          Upper: ExpandedSingleStreamBlocks [VL+SA | P] (2-way via p_tokens kwarg)
        """
        p = p_embs.contiguous()

        # Prepend time_token to SA only (physics stream is not diffusion-based)
        if has_time_token:
            sa = torch.cat([time_token, sa], dim=1)

        all_hidden = [sa]

        # ── RoPE for lower blocks: [VL | SA | P] ────────────────────────
        pe = None
        if self.use_rope:
            B = sa.shape[0]
            N_vl = vl.shape[1]
            N_sa = sa.shape[1]
            N_p = p.shape[1]
            device = sa.device

            total_len = N_vl + N_sa + N_p
            ids = torch.zeros(B, total_len, 2, dtype=torch.long, device=device)
            sa_start = N_vl

            # Sequence layout: [VL (N_vl) | SA (N_sa) | P (N_p)]
            p_start = N_vl + N_sa

            if self.positional_embeddings == "rope_sa_only":
                # SA positions on axis1
                current_idx = sa_start
                if has_time_token:
                    ids[:, current_idx : current_idx + self.num_temb_tokens, 1] = (
                        torch.arange(self.num_temb_tokens, device=device).unsqueeze(0).expand(B, -1)
                    )
                    current_idx += self.num_temb_tokens
                sa_len = N_sa - (self.num_temb_tokens if has_time_token else 0)
                start_pos = self.num_temb_tokens if has_time_token else 0
                ids[:, current_idx : current_idx + sa_len, 1] = (
                    torch.arange(start_pos, start_pos + sa_len, device=device)
                    .unsqueeze(0)
                    .expand(B, -1)
                )
                # P positions: axis0=1 (distinguish from SA), axis1=sequential
                ids[:, p_start:, 0] = 1
                ids[:, p_start:, 1] = torch.arange(N_p, device=device).unsqueeze(0).expand(B, -1)
            elif self.positional_embeddings == "rope_vl_sa":
                ids[:, :N_vl, 0] = torch.arange(N_vl, device=device).unsqueeze(0).expand(B, -1)
                # SA positions on axis1
                current_idx = sa_start
                if has_time_token:
                    ids[:, current_idx : current_idx + self.num_temb_tokens, 1] = (
                        torch.arange(self.num_temb_tokens, device=device).unsqueeze(0).expand(B, -1)
                    )
                    current_idx += self.num_temb_tokens
                sa_len = N_sa - (self.num_temb_tokens if has_time_token else 0)
                start_pos = (self.num_temb_tokens if has_time_token else 0) + 1
                ids[:, current_idx : current_idx + sa_len, 1] = (
                    torch.arange(start_pos, start_pos + sa_len, device=device)
                    .unsqueeze(0)
                    .expand(B, -1)
                )
                # P positions: axis0=1 (distinguish from SA/VL), axis1=sequential
                ids[:, p_start:, 0] = 1
                ids[:, p_start:, 1] = torch.arange(N_p, device=device).unsqueeze(0).expand(B, -1)

            pe = self.rope_embedder(ids)

        # ── Lower: Triple blocks [VL | SA | P] ───────────────────────────
        block_idx = 0
        for blk in self.double_blocks:
            sa, vl, p = blk(
                sa,
                vl,
                temb,
                pe=pe,
                shared_modulations=shared_modulations,
                has_time_token=has_time_token,
                block_idx=block_idx,
                encoder_attention_mask=encoder_attention_mask,
                p_tokens=p,
                physics_attention_mask=physics_attention_mask,
            )
            all_hidden.append(sa)
            block_idx += 1

        # Strip time tokens from SA
        if has_time_token:
            time_token = sa[:, : self.num_temb_tokens, :]
            sa = sa[:, self.num_temb_tokens :, :]

        # ── Upper: ExpandedSingleStreamBlocks with p_tokens ───────────────
        if len(self.single_blocks) > 0:
            vl_projected = self.vl_proj_to_sa(vl)
            N_vl_for_single = vl.shape[1]
            if has_time_token:
                x = torch.cat([vl_projected, time_token, sa], dim=1)
            else:
                x = torch.cat([vl_projected, sa], dim=1)

            # RoPE for upper blocks: [SA+VL (N_x) | P (N_p)]
            pe_single = None
            if self.use_rope:
                B_s = x.shape[0]
                N_x = x.shape[1]
                N_p = p.shape[1]
                ids_s = torch.zeros(B_s, N_x + N_p, 2, dtype=torch.long, device=x.device)

                if self.positional_embeddings in ("rope_sa_only", "rope_vl_sa"):
                    # SA+VL stream positions (same as standard SingleStreamBlock RoPE)
                    if self.positional_embeddings == "rope_vl_sa":
                        ids_s[:, :N_vl_for_single, 0] = (
                            torch.arange(N_vl_for_single, device=x.device)
                            .unsqueeze(0)
                            .expand(B_s, -1)
                        )
                    current_idx = N_vl_for_single
                    if has_time_token:
                        ids_s[:, current_idx : current_idx + self.num_temb_tokens, 1] = (
                            torch.arange(self.num_temb_tokens, device=x.device)
                            .unsqueeze(0)
                            .expand(B_s, -1)
                        )
                        current_idx += self.num_temb_tokens
                    sa_pure_len = sa.shape[1]
                    start_pos = (self.num_temb_tokens if has_time_token else 0) + 1
                    ids_s[:, current_idx : current_idx + sa_pure_len, 1] = (
                        torch.arange(start_pos, start_pos + sa_pure_len, device=x.device)
                        .unsqueeze(0)
                        .expand(B_s, -1)
                    )

                    # P stream positions: axis0=1 (distinguish from SA+VL), axis1=sequential
                    p_section_start = N_x
                    ids_s[:, p_section_start:, 0] = 1
                    ids_s[:, p_section_start:, 1] = (
                        torch.arange(N_p, device=x.device).unsqueeze(0).expand(B_s, -1)
                    )

                pe_single = self.rope_embedder(ids_s)

            # Build single-stream attention mask covering [VL+SA | P]
            single_attn_mask = None
            if encoder_attention_mask is not None or physics_attention_mask is not None:
                B_mask = x.shape[0]
                N_x_mask = x.shape[1]
                N_p_mask = p.shape[1]
                # VL+SA part
                if encoder_attention_mask is not None:
                    N_vl_mask = N_vl_for_single
                    rest_mask_x = torch.ones(
                        B_mask,
                        N_x_mask - N_vl_mask,
                        device=x.device,
                        dtype=encoder_attention_mask.dtype,
                    )
                    x_mask = torch.cat([encoder_attention_mask, rest_mask_x], dim=1)
                else:
                    x_mask = torch.ones(B_mask, N_x_mask, device=x.device, dtype=x.dtype)
                # P part
                if physics_attention_mask is not None:
                    p_mask = (
                        physics_attention_mask[:, None].expand(-1, N_p_mask).to(dtype=x_mask.dtype)
                    )
                else:
                    p_mask = torch.ones(B_mask, N_p_mask, device=x.device, dtype=x_mask.dtype)
                kv_mask = torch.cat([x_mask, p_mask], dim=1)
                single_attn_mask = kv_mask[:, None, None, :]
                single_attn_mask = torch.where(
                    single_attn_mask == 0,
                    torch.tensor(float("-inf"), device=x.device, dtype=x.dtype),
                    torch.tensor(0.0, device=x.device, dtype=x.dtype),
                )

            for blk in self.single_blocks:
                x, p = blk(
                    x,
                    temb,
                    pe=pe_single,
                    shared_modulation=shared_single_modulation,
                    time_token=time_token if has_time_token else None,
                    block_idx=block_idx,
                    p_tokens=p,
                    attn_mask=single_attn_mask,
                )
                block_idx += 1

            sa = x[:, -sa.shape[1] :, :]

        # ── Output projections ────────────────────────────────────────────
        action_out = self._output_projection(sa, temb)
        physics_out = self._output_projection_physics(p, temb)

        out = {"action": action_out, "physics": physics_out}
        if return_all_hidden_states:
            return out, all_hidden
        return out

    def _output_projection_physics(self, p, temb):
        """Physics output projection with AdaLN-zero style."""
        shift, scale = self.proj_out_physics_1(F.silu(temb)).chunk(2, dim=1)
        p = self.norm_out_physics(p) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_physics_2(p)

    def _output_projection(self, sa, temb):
        """Output projection with AdaLN-zero style."""
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        sa = self.norm_out(sa) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(sa)


class MSAT(JointBase):
    """
    Flux-style MSAT with DoubleStreamBlock + SingleStreamBlock architecture.
    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        depth_multi_stream: int = 12,  # Number of DoubleStreamBlocks (or TripleStreamBlocks)
        depth_single_stream: int = 0,  # Number of SingleStreamBlocks (Flux style)
        dropout: float = 0.1,
        attention_bias: Optional[bool] = None,  # If None, defaults to True
        activation_fn: str = "gelu-approximate",
        norm_eps: float = 1e-6,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        action_model_max_seq_len: int = 512,
        sa_dim: int = 1536,
        vl_dim: int = 1536,
        qk_norm: str = "none",
        use_swiglu: bool = False,
        mlp_ratio: float = 4.0,
        vl_mlp_ratio: Optional[
            float
        ] = None,  # If None, use mlp_ratio. Set lower to reduce VL stream params.
        temb_type: str = "layerwise_mod",  # "layerwise_mod", "shared_mod", or "input_token"
        remove_bias: bool = False,  # If True, remove bias from Modulation and projection layers
        pre_norm: str = "layer_norm",  # Pre-normalization type: "none", "layer_norm", or "rms_norm"
        post_norm: str = "none",  # Post-normalization type: "none", "layer_norm", or "rms_norm"
        rope_theta: float = 10000.0,  # Theta parameter for RoPE. Higher values result in slower rotation (smaller angles).
        # Physics (tactile/torque) conditioning
        use_physics: bool = False,
        physics_dim: int = 0,  # Total physics signal dimension (e.g. tactile_dim + torque_dim)
    ):
        super().__init__()
        self.use_physics = use_physics
        self.physics_dim = physics_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.timestep_encoder = TimestepEncoder(
            embedding_dim=self.inner_dim, compute_dtype=compute_dtype
        )
        self.positional_embeddings = positional_embeddings
        self.attention_head_dim = attention_head_dim
        self.temb_type = temb_type if temb_type is not None else "layerwise_mod"
        self.num_temb_tokens = 1  # Single time token when temb_type="input_token"

        # Set default attention_bias if not provided (MSAT default: True)
        if attention_bias is None:
            attention_bias = True

        # If remove_bias=True, override attention_bias to False
        if remove_bias:
            attention_bias = False
            _print(
                "[MSAT] remove_bias=True: overriding attention_bias to False for all attention layers"
            )

        # Create time token projection if temb_type is "input_token"
        # Time token is always added right before action tokens: [VL | S | time_token | A]
        if self.temb_type == "input_token":
            # Double stream: VL, SA - time_token goes before SA (which contains action)
            if self.inner_dim != sa_dim:
                self.time_token_proj = nn.Linear(self.inner_dim, sa_dim, bias=not remove_bias)
            else:
                self.time_token_proj = nn.Identity()

            self.time_token_pos_emb = None
        else:
            self.time_token_proj = None
            self.time_token_pos_emb = None

        self.freq_encoder = None
        self.freq_token_proj = None
        self.num_freq_tokens = 0

        # Create shared modulation modules if temb_type is "shared_mod"
        if self.temb_type == "shared_mod":
            # Double stream: VL, SA
            self.shared_sa_mod = Modulation(self.inner_dim, double=True, remove_bias=remove_bias)
            self.shared_vl_mod = Modulation(self.inner_dim, double=True, remove_bias=remove_bias)

            # For SingleStreamBlocks: use inner_dim as input (same as temb), then project output
            self.shared_single_mod = Modulation(
                self.inner_dim, double=False, remove_bias=remove_bias
            )
            if self.inner_dim != sa_dim:
                self.shared_single_mod_proj = nn.Linear(
                    self.inner_dim, sa_dim, bias=not remove_bias
                )
            else:
                self.shared_single_mod_proj = nn.Identity()

        _print("\nInitializing MSAT...")

        # Initialize RoPE embedder if needed
        if positional_embeddings == "rope_sa_only":
            # RoPE for SA stream only (attention_head_dim assumed to be 64 below)
            # Axis 0 (dim=16): 0 (unused)
            # Axis 1 (dim=48): SA sequence position
            _print(f"[MSAT] RoPE theta: {rope_theta}")
            self.rope_embedder = RoPEEmbedder1D(
                head_dim=attention_head_dim,
                axes_dim=[attention_head_dim // 4, attention_head_dim - attention_head_dim // 4],
                theta=rope_theta,
                max_seq_len=action_model_max_seq_len,
            )
            self.use_rope = True
        elif positional_embeddings == "rope_vl_sa":
            # RoPE for VL and SA streams (attention_head_dim assumed to be 64 below)
            # Axis 0 (dim=48): VL sequence position
            # Axis 1 (dim=16): SA sequence position
            self.rope_embedder = RoPEEmbedder1D(
                head_dim=attention_head_dim,
                axes_dim=[attention_head_dim - attention_head_dim // 4, attention_head_dim // 4],
                theta=rope_theta,
                max_seq_len=action_model_max_seq_len,
            )
            self.use_rope = True
        else:
            self.rope_embedder = None
            self.use_rope = False

        use_pos_emb = (
            positional_embeddings == "sinusoidal" and action_model_max_seq_len is not None
        ) or self.use_rope
        _print(
            f"[MSAT] 'positional_embeddings' of MSAT: {positional_embeddings}, "
            f"action_model_max_seq_len: {action_model_max_seq_len}, enabled: {use_pos_emb}"
        )

        self.sa_dim = sa_dim
        self.vl_dim = vl_dim

        # VL→SA projection (used by both physics and non-physics paths)
        if sa_dim != vl_dim:
            self.vl_proj_to_sa = nn.Linear(vl_dim, sa_dim, bias=not remove_bias)
            _print(f"[MSAT] Projecting VL dimension from {vl_dim} to {sa_dim}")
        else:
            self.vl_proj_to_sa = nn.Identity()

        if self.use_physics and physics_dim > 0:
            # ── Physics-enabled architecture ──────────────────────────────────────
            # Lower: ExpandedDoubleStreamBlocks [VL | SA | P] — extends DoubleStreamBlock with P stream
            # Upper: ExpandedSingleStreamBlocks [VL+SA | P]  — extends SingleStreamBlock with P stream
            # Pretrained weights load directly (same attribute names as base blocks).
            _print(f"\n[MSAT] Physics mode: use_physics=True, physics_dim={physics_dim}")
            _print(f"[MSAT] Lower: {depth_multi_stream} ExpandedDoubleStreamBlocks [VL | SA | P]")
            _print(f"[MSAT] Upper: {depth_single_stream} ExpandedSingleStreamBlocks [VL+SA | P]")

            self.double_blocks = self._build_expanded_double_blocks(
                depth=depth_multi_stream,
                sa_dim=sa_dim,
                vl_dim=vl_dim,
                p_dim=sa_dim,  # Physics tokens projected to sa_dim by PhysicalSignalEncoder
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                dropout=dropout,
                activation_fn=activation_fn,
                attention_bias=attention_bias,
                norm_eps=norm_eps,
                qk_norm=qk_norm,
                use_swiglu=use_swiglu,
                mlp_ratio=mlp_ratio,
                vl_mlp_ratio=vl_mlp_ratio,
                positional_embeddings=positional_embeddings,
                max_seq_length=action_model_max_seq_len,
                temb_type=self.temb_type,
                remove_bias=remove_bias,
                pre_norm=pre_norm,
                post_norm=post_norm,
            )

            single_stream_hidden_size = sa_dim
            self.single_blocks = self._build_expanded_single_blocks(
                depth=depth_single_stream,
                hidden_size=single_stream_hidden_size,
                p_dim=sa_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                dropout=dropout,
                activation_fn=activation_fn,
                attention_bias=attention_bias,
                norm_eps=norm_eps,
                qk_norm=qk_norm,
                use_swiglu=use_swiglu,
                mlp_ratio=mlp_ratio,
                positional_embeddings=positional_embeddings,
                max_seq_length=action_model_max_seq_len,
                temb_type=self.temb_type,
                remove_bias=remove_bias,
                pre_norm=pre_norm,
                post_norm=post_norm,
            )

            has_single_blocks = depth_single_stream > 0
            sa_hidden_dim = single_stream_hidden_size if has_single_blocks else sa_dim
            self._sa_hidden_dim = sa_hidden_dim

            # Physics output projection (AdaLN-zero style)
            self.norm_out_physics = nn.LayerNorm(sa_dim, elementwise_affine=False, eps=1e-6)
            self.proj_out_physics_1 = nn.Linear(self.inner_dim, 2 * sa_dim, bias=not remove_bias)
            self.proj_out_physics_2 = nn.Linear(sa_dim, output_dim, bias=not remove_bias)

        else:
            # ── Standard architecture (no physics) ────────────────────────────────
            # Lower: DoubleStreamBlocks [VL | SA]
            # Upper: SingleStreamBlocks [VL_proj | time_token | SA]

            self.double_blocks = self._build_double_blocks(
                depth=depth_multi_stream,
                sa_dim=sa_dim,
                vl_dim=vl_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                dropout=dropout,
                activation_fn=activation_fn,
                attention_bias=attention_bias,
                norm_eps=norm_eps,
                qk_norm=qk_norm,
                use_swiglu=use_swiglu,
                mlp_ratio=mlp_ratio,
                vl_mlp_ratio=vl_mlp_ratio,
                positional_embeddings=positional_embeddings,
                max_seq_length=action_model_max_seq_len,
                temb_type=self.temb_type,
                remove_bias=remove_bias,
                pre_norm=pre_norm,
                post_norm=post_norm,
            )

            single_stream_hidden_size = sa_dim
            self.single_blocks = self._build_single_blocks(
                depth=depth_single_stream,
                hidden_size=single_stream_hidden_size,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                dropout=dropout,
                activation_fn=activation_fn,
                attention_bias=attention_bias,
                norm_eps=norm_eps,
                qk_norm=qk_norm,
                use_swiglu=use_swiglu,
                mlp_ratio=mlp_ratio,
                positional_embeddings=positional_embeddings,
                max_seq_length=action_model_max_seq_len,
                temb_type=self.temb_type,
                remove_bias=remove_bias,
                pre_norm=pre_norm,
                post_norm=post_norm,
            )

            has_single_blocks = depth_single_stream > 0
            sa_hidden_dim = single_stream_hidden_size if has_single_blocks else sa_dim
            self._sa_hidden_dim = sa_hidden_dim

        # Output projection (AdaLN-zero style) - for action prediction
        self.norm_out = nn.LayerNorm(sa_hidden_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * sa_hidden_dim, bias=not remove_bias)
        self.proj_out_2 = nn.Linear(sa_hidden_dim, output_dim, bias=not remove_bias)
        _print(
            f"[MSAT] Output projection: sa_hidden_dim={sa_hidden_dim} -> output_dim={output_dim}"
        )

        self._remove_bias = remove_bias

        _print(
            "[MSAT] Total number of MSAT parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # SA tokens
        encoder_hidden_states: torch.Tensor,  # VL tokens
        timestep: Optional[torch.LongTensor] = None,
        return_all_hidden_states: bool = False,
        encoder_attention_mask: Optional[
            torch.Tensor
        ] = None,  # [B, N_vl] VL attention mask (1=visible, 0=masked)
        physics_embs: Optional[
            torch.Tensor
        ] = None,  # [B, N_p, sa_dim] Physics tokens (when use_physics=True)
        physics_attention_mask: Optional[
            torch.Tensor
        ] = None,  # [B] per-sample physics mask (1=visible, 0=masked)
    ):
        return self._forward_inner(
            hidden_states,
            encoder_hidden_states,
            timestep,
            return_all_hidden_states,
            encoder_attention_mask,
            physics_embs,
            physics_attention_mask,
        )
