"""Graph-safe wrappers for Qwen3VL Vision Model.

Pre-computes data-dependent values (pos_embeds, rotary, cu_seqlens) in __init__,
then uses them as static values in forward.  Replaces the original
Qwen3VLVisionModel and Qwen3VLVisionAttention for CUDA Graph / compile / TRT.

Motion-module support (RLDX-1 midtrain variants):
  When the visual model has a motion_block, inserts it after
  motion_insert_layer with a pre-computed static grid_sizes buffer.
  Only supports "vision_encoder" injection mode (residual add).
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from rldx.utils.dist import rank_zero_print as _print


# Attention


class GraphSafeQwen3VLVisionAttention(nn.Module):
    """Qwen3VLVisionAttention with pre-computed static lengths/max_seqlen.

    Eliminates .tolist() and .max() calls that cause graph breaks.
    """

    def __init__(self, attn, static_lengths, static_max_seqlen):
        super().__init__()
        self.qkv = attn.qkv
        self.proj = attn.proj
        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dim
        self.scaling = attn.scaling
        self.config = attn.config
        self.attention_dropout = attn.attention_dropout
        self.is_causal = attn.is_causal
        self.num_key_value_groups = attn.num_key_value_groups
        self.static_lengths = static_lengths
        self.static_max_seqlen = static_max_seqlen

        # Resolve all attn functions at init; select at runtime via config reference
        _vis_mod = sys.modules[type(attn).__module__]
        self._apply_rope_vision = getattr(_vis_mod, "apply_rotary_pos_emb_vision")
        _all_attn_fns = getattr(_vis_mod, "ALL_ATTENTION_FUNCTIONS")
        self._fa2_fn = _all_attn_fns.get("flash_attention_2")
        self._sdpa_fn = _all_attn_fns.get("sdpa")
        self._eager_fn = getattr(_vis_mod, "eager_attention_forward")

    @property
    def _use_fa2(self):
        return self.config._attn_implementation == "flash_attention_2"

    @property
    def _attn_fn(self):
        impl = self.config._attn_implementation
        if impl == "flash_attention_2":
            return self._fa2_fn
        elif impl == "sdpa":
            return self._sdpa_fn
        else:
            return self._eager_fn

    def forward(
        self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs
    ):
        seq_length = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        q, k = self._apply_rope_vision(q, k, cos, sin)

        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        if self._use_fa2:
            attn_output, _ = self._attn_fn(
                self,
                q,
                k,
                v,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=self.static_max_seqlen,
                max_length_k=self.static_max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            splits = [torch.split(t, self.static_lengths, dim=2) for t in (q, k, v)]
            attn_output = torch.cat(
                [
                    self._attn_fn(
                        self,
                        qi,
                        ki,
                        vi,
                        attention_mask=None,
                        scaling=self.scaling,
                        dropout=0.0,
                        is_causal=False,
                        **kwargs,
                    )[0]
                    for qi, ki, vi in zip(*splits)
                ],
                dim=1,
            )

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)


# Vision Model


class GraphSafeQwen3VLVisionModel(nn.Module):
    """Qwen3VLVisionModel with pre-computed static buffers.

    Data-dependent operations replaced:
      - fast_pos_embed_interpolate(grid_thw) → self.pos_embeds
      - rot_pos_emb(grid_thw) → self.pos_cos, self.pos_sin
      - cu_seqlens computation → self.cu_seqlens
      - VisionAttention splits → GraphSafeQwen3VLVisionAttention
      - get_image_features split_sizes → self.split_sizes
    """

    def __init__(self, visual, grid_thw, num_frames=1, num_views=1):
        super().__init__()
        self._visual = visual
        grid_thw = grid_thw.reshape(-1, 3) if grid_thw.ndim == 3 else grid_thw

        with torch.no_grad():
            # Static variables
            self.pos_embeds = visual.fast_pos_embed_interpolate(grid_thw)

            rotary = visual.rot_pos_emb(grid_thw)
            emb = torch.cat((rotary, rotary), dim=-1)
            self.pos_cos = emb.cos()
            self.pos_sin = emb.sin()

            cu = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
                dim=0, dtype=torch.int32
            )
            self.cu_seqlens = F.pad(cu, (1, 0), value=0)

            lengths = (self.cu_seqlens[1:] - self.cu_seqlens[:-1]).tolist()
            self.max_seqlen = max(lengths)
            self.split_sizes = (grid_thw.prod(-1) // visual.spatial_merge_size**2).tolist()

        # Wrap attention blocks
        for blk in visual.blocks:
            blk.attn = GraphSafeQwen3VLVisionAttention(blk.attn, lengths, self.max_seqlen)

        # --- Motion-module setup ---
        self.motion_block = getattr(visual, "motion_block", None)
        self.motion_insert_layer = getattr(visual, "motion_insert_layer", None)

        if self.motion_block is not None:
            # Force eval for graph-safety (BatchNorm uses stored running stats)
            self.motion_block.eval()

            # Pre-compute static grid info for the motion module
            num_images = grid_thw.shape[0]
            true_batch = num_images // (num_frames * num_views)
            h = grid_thw[0, 1].item()
            w = grid_thw[0, 2].item()
            num_patches = h * w

            # Patch motion-module internals: replace all grid_sizes GPU reads with Python ints.
            # The motion-module forward does grid_sizes[0] → GPU→CPU sync → breaks CUDA graph.
            _static_grid = (num_frames, h, w)  # Python ints, closure-captured

            # STSSTransformation: uses `t, h, w = grid_sizes[0]`
            for enc in self.motion_block.stss_encoders:
                _orig_stss_trans = enc.stss_transformation
                _orig_stss_fwd = _orig_stss_trans.forward

                def _make_stss_trans_fwd(orig, grid=_static_grid):
                    def fwd(x, grid_sizes):
                        # Replace grid_sizes arg with static Python ints
                        return orig(x, [grid])

                    return fwd

                enc.stss_transformation.forward = _make_stss_trans_fwd(_orig_stss_fwd)

                # STSSEncoder: uses `t, h, w = grid_sizes[0]`
                _orig_enc_fwd = enc.forward

                def _make_enc_fwd(orig_enc, grid=_static_grid):
                    def fwd(x, grid_sizes=None):
                        return orig_enc(x, grid_sizes=[grid])

                    return fwd

                enc.forward = _make_enc_fwd(_orig_enc_fwd)

            # MotionBlock: skip `all_same_grid` CPU check + gradient hook
            _orig_encoders = self.motion_block.stss_encoders
            _orig_use_ls = self.motion_block.use_layerscale
            if _orig_use_ls:
                _orig_ls = self.motion_block.layerscale
            else:
                _orig_out_proj = self.motion_block.out_proj

            def _motion_forward_static(x, grid_sizes):
                out = x
                encoder_outputs = []
                for enc in _orig_encoders:
                    out = enc(out, grid_sizes=grid_sizes)
                    encoder_outputs.append(out)
                out = torch.stack(encoder_outputs, dim=0).sum(dim=0)
                if _orig_use_ls:
                    return out * _orig_ls
                return _orig_out_proj(out)

            self.motion_block.forward = _motion_forward_static

            self._motion_true_batch = true_batch
            self._motion_num_frames = num_frames
            self._motion_num_views = num_views
            self._motion_h = h
            self._motion_w = w
            self._motion_num_patches = num_patches

            # Static grid_sizes for MotionBlock: (B*V, 3) = [[T, H, W], ...]
            motion_grid_sizes = torch.tensor(
                [[num_frames, h, w]] * (true_batch * num_views),
                dtype=torch.long,
                device=grid_thw.device,
            )
            self.register_buffer("motion_grid_sizes", motion_grid_sizes)

            injection_point = getattr(visual, "motion_injection_point", "vision_encoder")
            _print(
                f"  [Motion] insert_layer={self.motion_insert_layer}, "
                f"injection={injection_point}, "
                f"batch={true_batch}, T={num_frames}, V={num_views}, "
                f"H={h}, W={w}"
            )
        else:
            self.motion_grid_sizes = None

        _print(
            f"  Static buffers (vision): pos_embeds={list(self.pos_embeds.shape)}, "
            f"cos={list(self.pos_cos.shape)}, cu_seqlens={list(self.cu_seqlens.shape)}, "
            f"lengths={lengths}, max_seqlen={self.max_seqlen}, "
            f"split_sizes={self.split_sizes}"
        )

    def _apply_motion_static(self, hidden_states):
        """Apply the motion module with pre-computed static grid sizes (vision_encoder mode).

        Reshapes flat tokens → (B, V, T, P, D) for MotionBlock, then adds residual.
        Includes raster↔interleaved patch order conversion (n1.6-0407, fcc08ad).
        """
        B = self._motion_true_batch
        T = self._motion_num_frames
        V = self._motion_num_views
        P = self._motion_num_patches
        D = hidden_states.shape[-1]
        h = self._motion_h
        w = self._motion_w
        merge_size = self._visual.spatial_merge_size
        merged_h, merged_w = h // merge_size, w // merge_size

        # (num_images * P, D) -> (B, T, V, P, D)
        hidden_3d = hidden_states.reshape(B * T * V, P, D)
        hidden_5d = hidden_3d.reshape(B, T, V, P, D)

        # Undo Qwen's block-interleaved patch ordering before the motion module.
        # Tokens are in (merged_h, merged_w, merge_size, merge_size) order;
        # the motion module expects raster (h, w) order for its Conv3d /
        # correlation ops.
        hidden_5d = hidden_5d.reshape(B, T, V, merged_h, merged_w, merge_size, merge_size, D)
        hidden_5d = hidden_5d.permute(0, 1, 2, 3, 5, 4, 6, 7).contiguous()
        hidden_5d = hidden_5d.reshape(B, T, V, P, D)

        # (B, T, V, P, D) -> (B, V, T, P, D) -> (B*V*T*P, D)
        hidden_bvtpd = hidden_5d.permute(0, 2, 1, 3, 4).contiguous()
        motion_input = hidden_bvtpd.reshape(B * V * T * P, D)

        motion_out = self.motion_block(motion_input, self.motion_grid_sizes)

        # (B*V*T*P, D) -> (B, V, T, P, D) -> (B, T, V, P, D)
        motion_out = motion_out.reshape(B, V, T, P, D)
        motion_out = motion_out.permute(0, 2, 1, 3, 4).contiguous()

        # Convert motion-module output back to block-interleaved order for residual addition
        motion_out = motion_out.reshape(B, T, V, merged_h, merge_size, merged_w, merge_size, D)
        motion_out = motion_out.permute(0, 1, 2, 3, 5, 4, 6, 7).contiguous()
        motion_out = motion_out.reshape(B, T, V, P, D)

        return hidden_states + motion_out.reshape(-1, D)

    def forward(self, hidden_states, grid_thw=None, **kwargs):
        hidden_states = self._visual.patch_embed(hidden_states)
        hidden_states = hidden_states + self.pos_embeds

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self._visual.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=self.cu_seqlens,
                position_embeddings=(self.pos_cos, self.pos_sin),
                **kwargs,
            )

            # Motion-module insertion after the configured layer.
            if self.motion_block is not None and layer_num == self.motion_insert_layer:
                hidden_states = self._apply_motion_static(hidden_states)

            if layer_num in self._visual.deepstack_visual_indexes:
                idx = self._visual.deepstack_visual_indexes.index(layer_num)
                deepstack_feature_lists.append(
                    self._visual.deepstack_merger_list[idx](hidden_states)
                )

        hidden_states = self._visual.merger(hidden_states)
        return hidden_states, deepstack_feature_lists

    def __getattr__(self, name):
        """Delegate attribute access to the original visual model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._visual, name)
