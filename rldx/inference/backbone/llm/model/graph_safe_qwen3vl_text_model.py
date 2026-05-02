"""Graph-safe wrapper for Qwen3VL Text Model (LLM decoder).

Pre-computes FA kwargs and handles LayerWrapper compression with static indices.
Eliminates all data-dependent graph breaks (.item(), torch.nonzero, etc.).

Supports two attention modes:
  - "flash_attention_2": passes cu_seq_lens/max_length FA kwargs (default, eager)
  - "sdpa": omits FA kwargs, relies on SDPA is_causal (for ONNX/TRT export)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from rldx.utils.dist import rank_zero_print as _print


def _compute_fa_kwargs(seq_len, device):
    """Compute static flash_attention varlen kwargs for a single contiguous sequence."""
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    return cu, int(seq_len)


def _find_compress_info(language_model, input_ids, n_cog_tokens, num_views=None):
    """Find compression layer and compute static begin/end indices.

    Args:
        num_views: number of camera views. When num_views >= 2, vanilla
            LayerWrapper keeps the last (num_views - 1) image sets uncompressed.
            If begin == end, compression is skipped (compress_mask=False).
    """
    for idx, layer in enumerate(language_model.layers):
        if (
            hasattr(layer, "layer")
            and hasattr(layer, "internal_projection")
            and layer.layer_idx == layer.internal_projection
        ):
            with torch.no_grad():
                dummy = torch.zeros(1, input_ids.shape[1], 1, device=input_ids.device)
                begin_idx, end_idx = layer.get_removing_indices(
                    dummy, input_ids, num_views=num_views
                )
            b = begin_idx[0, 0].item()
            e = end_idx[0, 0].item()
            if b >= e:
                # compress_mask=False in vanilla → no compression
                return None
            L_llm = input_ids.shape[1] + n_cog_tokens
            L_out = b + 1 + (L_llm - e)
            return {
                "compress_layer_idx": idx,
                "static_begin": b,
                "static_end": e,
                "static_out_len": L_out,
            }
    return None


class GraphSafeQwen3VLTextModel(nn.Module):
    """Qwen3VLTextModel with graph-safe forward.

    Data-dependent operations replaced:
      - prepare_fa_kwargs_from_position_ids (.item()) → pre-computed FA kwargs
      - LayerWrapper compression (torch.nonzero) → static begin/end slice
      - create_causal_mask → attention_mask=None (FA varlen uses cu_seqlens)

    Attention modes:
      - "flash_attention_2": FA varlen params (cu_seqlens, max_length) — eager path
      - "sdpa": no FA params, SDPA uses is_causal=True — TRT/ONNX export path
    """

    def __init__(
        self, text_model, input_ids, n_cog_tokens=0, attn_impl="flash_attention_2", num_views=None
    ):
        super().__init__()
        self._text_model = text_model
        self.attn_impl = attn_impl

        L_ids = input_ids.shape[1]
        device = input_ids.device
        L_pre = L_ids + n_cog_tokens

        # Compression info (num_views affects which image tokens are compressed)
        self.compress_info = _find_compress_info(
            text_model, input_ids, n_cog_tokens, num_views=num_views
        )

        if self.compress_info is not None:
            ci = self.compress_info
            L_post = ci["static_out_len"]
            _print(
                f"  Static buffers (compression): layer_idx={ci['compress_layer_idx']}, "
                f"begin={ci['static_begin']}, end={ci['static_end']}, "
                f"L_llm={L_pre} → {L_post} (L_ids={L_ids}, cog_tokens={n_cog_tokens})"
            )
        else:
            L_post = L_pre

        # Static FA varlen params (Python int + tensor — no .item() at runtime)
        self.pre_cu_seqlens, self.pre_max_seqlen = _compute_fa_kwargs(L_pre, device)
        self.post_cu_seqlens, self.post_max_seqlen = _compute_fa_kwargs(L_post, device)

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        position_ids=None,
        position_embeddings=None,
        deepstack_add=None,
        attention_mask=None,
        use_cache=None,
        cache_position=None,
        **kwargs,
    ):
        """Graph-safe forward.

        Args:
            position_embeddings: Optional pre-computed (cos, sin) tuple.
                If provided, skips RoPE computation (for ONNX export).
            deepstack_add: Optional [num_ds, B, L, D] tensor of additive
                DeepStack features (added after the first num_ds layers).
        """
        tm = self._text_model

        if inputs_embeds is None:
            inputs_embeds = tm.embed_tokens(input_ids)

        # 2D→3D MROPE expansion (matches original Qwen3VLTextModel)
        if position_ids is not None and position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # Cache for engine access (engines read after one forward pass)
        self._cached_inputs_embeds = inputs_embeds.detach()
        self._cached_position_ids = position_ids.detach() if position_ids is not None else None

        hidden_states = inputs_embeds
        if position_embeddings is None:
            position_embeddings = tm.rotary_emb(hidden_states, position_ids)

        ci = self.compress_info
        cu_seqlens = self.pre_cu_seqlens
        max_seqlen = self.pre_max_seqlen
        use_fa = self.attn_impl == "flash_attention_2"

        for idx, layer in enumerate(tm.layers):
            inner = layer.layer if hasattr(layer, "layer") else layer

            if ci is not None and idx == ci["compress_layer_idx"]:
                # Static compression: replace image tokens with mean motion token
                # Use masked sum to match vanilla LayerWrapper's accumulation order
                b, e = ci["static_begin"], ci["static_end"]
                n_drop = e - b
                drop_mask = torch.zeros(
                    1,
                    hidden_states.shape[1],
                    1,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                drop_mask[:, b:e, :] = 1.0
                motion = (hidden_states * drop_mask).sum(dim=1, keepdim=True) / n_drop
                front = hidden_states[:, :b, :]
                back = hidden_states[:, e:, :]
                hidden_states = torch.cat([front, motion, back], dim=1)

                # Compress position_embeddings
                cos, sin = position_embeddings
                cos = torch.cat([cos[:, :b], cos[:, b : b + 1], cos[:, e:]], dim=1)
                sin = torch.cat([sin[:, :b], sin[:, b : b + 1], sin[:, e:]], dim=1)
                position_embeddings = (cos, sin)

                # Switch to post-compression FA params
                cu_seqlens = self.post_cu_seqlens
                max_seqlen = self.post_max_seqlen

            if use_fa:
                hidden_states = inner(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=None,
                    cu_seq_lens_q=cu_seqlens,
                    cu_seq_lens_k=cu_seqlens,
                    max_length_q=max_seqlen,
                    max_length_k=max_seqlen,
                )
            else:
                hidden_states = inner(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=None,
                )
            # Unwrap tuple output from decoder layer
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

            # DeepStack additive features
            if deepstack_add is not None and idx < deepstack_add.shape[0]:
                hidden_states = hidden_states + deepstack_add[idx]

        hidden_states = tm.norm(hidden_states)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states)

    def __getattr__(self, name):
        """Delegate attribute access to the original text model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._text_model, name)
