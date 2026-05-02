"""Vision Encoder chain for VLM.

CustomVisionEncoderChain:
  Chains N VisionBlocks + merger (+ deepstack mergers) into a single compilable nn.Module.

  Attention: torch.ops.rldx_backbone.vision_attention — fused Triton kernel that reads
  Q/K/V directly from the fused QKV buffer, applies baked RoPE in-register,
  and runs non-causal varlen attention with per-image boundaries from
  cu_seqlens. Single op replaces QKV reshape + Python RoPE + HF wrapper.
  VLA inference has fixed sequence length, so cos/sin are static constants.
  rotate_half sign is pre-baked into sin at init time.

  Norm and MLP use original module references for exact behavior match.

Qwen3VLVisionBlock structure (per block):
  - norm1: LayerNorm(D, bias=True)
  - attn:
      - qkv: Linear(D, 3*D, bias=True)    ← 1 fused QKV
      - RoPE + non-causal attention: fused via torch.ops.rldx_backbone.vision_attention
      - proj: Linear(D, D)                ← O projection
  - norm2: LayerNorm(D, bias=True)
  - mlp:
      - linear_fc1: Linear(D, intermediate, bias=True)
      - act_fn (original module)
      - linear_fc2: Linear(intermediate, D, bias=True)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionLayerParam(nn.Module):
    """Per-block weight container for Qwen3VL VisionBlock.

    Keeps original norm/MLP modules for exact behavior match.
    Attention weights stored as buffers for baked RoPE path.
    """

    def __init__(self, block):
        super().__init__()
        # Original modules (exact behavior match)
        self.norm1 = block.norm1
        self.norm2 = block.norm2
        self.mlp = block.mlp

        # Attention weights as buffers (for baked RoPE application)
        attn = block.attn
        self.register_buffer("qkv_weight", attn.qkv.weight.data)
        self.register_buffer("qkv_bias", attn.qkv.bias.data)
        self.register_buffer("proj_weight", attn.proj.weight.data)
        if attn.proj.bias is not None:
            self.register_buffer("proj_bias", attn.proj.bias.data)
        else:
            self.proj_bias = None
        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dim
        self.scaling = attn.scaling


class CustomVisionEncoderChain(nn.Module):
    """Wraps Qwen3VL VisionBlocks + merger into a single compilable chain.

    Attention uses baked RoPE + HF attention wrapper (torch.compile compatible).

    Args:
        blocks: list/ModuleList of VisionBlock modules
        merger: VisionPatchMerger
        deepstack_merger_list: ModuleList of deepstack mergers
        deepstack_visual_indexes: list[int]
        pos_cos, pos_sin: pre-computed RoPE (M, head_dim)
        cu_seqlens: cumulative sequence lengths
        max_seqlen: max sequence length
    """

    def __init__(
        self,
        blocks,
        merger,
        deepstack_merger_list,
        deepstack_visual_indexes,
        pos_cos,
        pos_sin,
        cu_seqlens,
        max_seqlen,
        motion_block=None,
        motion_insert_layer=None,
        motion_grid_sizes=None,
        enable_motion_fast=True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([VisionLayerParam(b) for b in blocks])
        self.blocks = blocks  # for forward_with_intermediates fallback
        self.merger = merger
        self.deepstack_merger_list = deepstack_merger_list
        self.deepstack_visual_indexes = list(deepstack_visual_indexes)
        self.n_layers = len(self.layers)

        # Bake RoPE: fold rotate_half sign into sin
        #   d < D/2: rope_sin[d] = -sin[d]  (rotate_half negates first half)
        #   d >= D/2: rope_sin[d] = +sin[d]
        # Kernel expects (M, head_dim) — no broadcast unsqueeze needed.
        D = pos_cos.shape[-1]
        half = D // 2
        rope_sin = pos_sin.clone()
        rope_sin[:, :half] = -rope_sin[:, :half]

        self.register_buffer("rope_cos", pos_cos.contiguous())
        self.register_buffer("rope_sin", rope_sin.contiguous())

        # cu_seqlens (shared with fused op); originals kept for forward_with_intermediates
        self.register_buffer("pos_cos", pos_cos)
        self.register_buffer("pos_sin", pos_sin)
        self.register_buffer("cu_seqlens", cu_seqlens)
        self.max_seqlen = max_seqlen

        # Motion (v1.4-all): inserted after motion_insert_layer
        # MotionBlock expects (B*V*T*P, D) with grid_sizes=(B*V, 3)
        # Vision encoder hidden_states are (B*T*V*P, D) — need T↔V permute
        self.motion_block = motion_block
        self.motion_insert_layer = motion_insert_layer
        self.enable_motion_fast = enable_motion_fast
        if motion_grid_sizes is not None:
            self.register_buffer("motion_grid_sizes", motion_grid_sizes)
            # Pre-compute reshape dims from grid_sizes
            num_entries = motion_grid_sizes.shape[0]  # B*V
            t, h, w = motion_grid_sizes[0].tolist()
            self._motion_num_views = num_entries  # B*V (B=1 in deployment)
            self._motion_num_frames = t
            self._motion_num_patches = h * w
            # For raster↔interleaved conversion (n1.6-0407)
            self._motion_h = h
            self._motion_w = w
            self._motion_merge_size = 2  # Qwen3-VL default spatial_merge_size
            mh, mw = (
                self._motion_h // self._motion_merge_size,
                self._motion_w // self._motion_merge_size,
            )
            base = torch.arange(
                self._motion_num_frames * self._motion_num_views * self._motion_num_patches,
                device=motion_grid_sizes.device,
            ).reshape(
                self._motion_num_frames,
                self._motion_num_views,
                mh,
                mw,
                self._motion_merge_size,
                self._motion_merge_size,
            )
            motion_input_order = base.permute(1, 0, 2, 4, 3, 5).reshape(-1)
            self.register_buffer("_motion_input_order", motion_input_order)
            self.register_buffer("_motion_restore_order", torch.argsort(motion_input_order))
        else:
            self.motion_grid_sizes = None

    def _apply_motion_fast(self, hidden_states):
        if not self.enable_motion_fast:
            return None
        tokens_per_group = self._motion_input_order.numel()
        total_tokens = hidden_states.shape[0]
        if total_tokens % tokens_per_group != 0:
            return None

        groups = total_tokens // tokens_per_group
        if groups == 1:
            motion_in = hidden_states.index_select(0, self._motion_input_order)
            motion_out = self.motion_block(motion_in, self.motion_grid_sizes)
            return motion_out.index_select(0, self._motion_restore_order)

        offsets = torch.arange(groups, device=hidden_states.device)[:, None] * tokens_per_group
        gather_idx = (offsets + self._motion_input_order.unsqueeze(0)).reshape(-1)
        restore_idx = (offsets + self._motion_restore_order.unsqueeze(0)).reshape(-1)
        motion_in = hidden_states.index_select(0, gather_idx)
        motion_out = self.motion_block(motion_in, self.motion_grid_sizes)
        return motion_out.index_select(0, restore_idx)

    def forward(self, hidden_states):
        """Run VisionBlocks + merger with fused RoPE + non-causal varlen attention.

        Args:
            hidden_states: (M, D) bf16 — output of patch_embed + pos_embed

        Returns:
            merged: (M', D_out) bf16
            deepstack_features: list of (M', D_out) bf16
        """
        from .kernels.fused_add2_layernorm import forward as fused_add2_ln

        deepstack_features = []
        n_layers = self.n_layers

        # First layer's pre-attention LayerNorm
        normed = self.layers[0].norm1(hidden_states)

        for i, layer in enumerate(self.layers):
            # 1. (normed is pre-computed from previous epilogue or first-layer norm)

            # 2. QKV projection → fused (M, 3 * num_heads * head_dim)
            qkv = F.linear(normed, layer.qkv_weight, layer.qkv_bias)

            # 3. Fused attention: baked RoPE + non-causal varlen attention
            #    Kernel reads Q/K/V directly from qkv (no permute/reshape copy).
            attn_out = torch.ops.rldx_backbone.vision_attention(
                qkv,
                self.rope_cos,
                self.rope_sin,
                self.cu_seqlens,
                layer.scaling,
                layer.num_heads,
                layer.head_dim,
            )  # (M, num_heads * head_dim)

            # 4. O projection + post-attention epilogue: residual + norm2
            if layer.proj_bias is not None:
                attn_out = F.linear(attn_out, layer.proj_weight, layer.proj_bias)
            else:
                attn_out = F.linear(attn_out, layer.proj_weight)
            hidden_states, norm2_out = fused_add2_ln(
                attn_out, hidden_states, layer.norm2.weight, layer.norm2.bias
            )

            # 5. MLP + post-MLP epilogue: residual + next layer's norm1
            mlp_out = layer.mlp(norm2_out)
            if i < n_layers - 1:
                next_layer = self.layers[i + 1]
                hidden_states, normed = fused_add2_ln(
                    mlp_out, hidden_states, next_layer.norm1.weight, next_layer.norm1.bias
                )
            else:
                hidden_states = hidden_states + mlp_out

            # Motion insertion (v1.4-all): after motion_insert_layer
            # hidden_states is (B*T*V*P, D) but MotionBlock expects (B*V*T*P, D)
            # Includes raster↔interleaved patch order conversion (n1.6-0407)
            if self.motion_block is not None and i == self.motion_insert_layer:
                motion_fast = self._apply_motion_fast(hidden_states)
                if motion_fast is None:
                    BV, T, P = (
                        self._motion_num_views,
                        self._motion_num_frames,
                        self._motion_num_patches,
                    )
                    D = hidden_states.shape[-1]
                    ms = self._motion_merge_size
                    mh, mw = self._motion_h // ms, self._motion_w // ms

                    motion_in = hidden_states.reshape(-1, T, BV, P, D)
                    motion_in = motion_in.reshape(-1, T, BV, mh, mw, ms, ms, D)
                    motion_in = motion_in.permute(0, 1, 2, 3, 5, 4, 6, 7).contiguous()
                    motion_in = motion_in.reshape(-1, T, BV, P, D)
                    motion_in = motion_in.permute(0, 2, 1, 3, 4).contiguous()
                    motion_out = self.motion_block(motion_in.reshape(-1, D), self.motion_grid_sizes)
                    motion_out = (
                        motion_out.reshape(-1, BV, T, P, D).permute(0, 2, 1, 3, 4).contiguous()
                    )
                    motion_out = motion_out.reshape(-1, T, BV, mh, ms, mw, ms, D)
                    motion_out = motion_out.permute(0, 1, 2, 3, 5, 4, 6, 7).contiguous()
                    motion_out = motion_out.reshape(-1, T, BV, P, D)
                    motion_fast = motion_out.reshape(-1, D)
                hidden_states = hidden_states + motion_fast
                # Re-compute normed for next layer since hidden_states changed
                if i < n_layers - 1:
                    normed = next_layer.norm1(hidden_states)

            # DeepStack
            if i in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(i)
                deepstack_features.append(self.deepstack_merger_list[idx](hidden_states))

        merged = self.merger(hidden_states)
        return merged, deepstack_features

    def forward_with_intermediates(self, hidden_states):
        """Forward with per-block intermediate capture (uses original blocks)."""
        deepstack_features = []
        intermediates = []

        for i, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=self.cu_seqlens,
                position_embeddings=(self.pos_cos, self.pos_sin),
            )
            if self.motion_block is not None and i == self.motion_insert_layer:
                BV, T, P = self._motion_num_views, self._motion_num_frames, self._motion_num_patches
                D = hidden_states.shape[-1]
                ms = self._motion_merge_size
                mh, mw = self._motion_h // ms, self._motion_w // ms
                motion_in = hidden_states.reshape(-1, T, BV, P, D)
                motion_in = motion_in.reshape(-1, T, BV, mh, mw, ms, ms, D)
                motion_in = motion_in.permute(0, 1, 2, 3, 5, 4, 6, 7).contiguous()
                motion_in = motion_in.reshape(-1, T, BV, P, D)
                motion_in = motion_in.permute(0, 2, 1, 3, 4).contiguous()
                motion_out = self.motion_block(motion_in.reshape(-1, D), self.motion_grid_sizes)
                motion_out = motion_out.reshape(-1, BV, T, P, D).permute(0, 2, 1, 3, 4).contiguous()
                motion_out = motion_out.reshape(-1, T, BV, mh, ms, mw, ms, D)
                motion_out = motion_out.permute(0, 1, 2, 3, 5, 4, 6, 7).contiguous()
                motion_out = motion_out.reshape(-1, T, BV, P, D)
                hidden_states = hidden_states + motion_out.reshape(-1, D)
            intermediates.append(
                {
                    "hidden_states": hidden_states.detach().clone(),
                }
            )
            if i in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(i)
                deepstack_features.append(self.deepstack_merger_list[idx](hidden_states))

        merged = self.merger(hidden_states)
        return merged, deepstack_features, intermediates
