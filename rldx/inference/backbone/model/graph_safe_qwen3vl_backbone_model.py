"""Unified graph-safe VLM: Vision + Glue + LLM + Projection.

Combines GraphSafeQwen3VLVisionModel and GraphSafeQwen3VLTextModel with the
embedding/scatter/cog-token glue into a single nn.Module. All data-dependent
operations are pre-computed in __init__; forward() is a pure computation graph.

Engine builders (TRT, CustomVLMChain) receive this model and decompose it via
the public gs_visual / gs_text attributes for engine-specific optimization.

The cog-token slot is appended after the language IDs and routed through
the LLM as additional tokens whose embeddings come from a learned
``backbone.cog_emb`` parameter.
"""

from __future__ import annotations

from llm.model.graph_safe_qwen3vl_text_model import GraphSafeQwen3VLTextModel
import torch
import torch.nn as nn
from vision_encoder.model.graph_safe_qwen3vl_vision_model import GraphSafeQwen3VLVisionModel

from rldx.utils.dist import rank_zero_print as _print


class GraphSafeQwen3VLBackbone(nn.Module):
    """Unified graph-safe VLM model.

    Forward flow:
      pixel_values → Vision Encoder → Embed + Image Scatter →
      (cog-token append) → LLM Decoder (with compression) →
      (cog-token extract) → Projection → backbone_features

    Public attributes for engine builders:
      gs_visual:    GraphSafeQwen3VLVisionModel (vision static buffers)
      gs_text:      GraphSafeQwen3VLTextModel   (LLM static buffers)
      embed_tokens: nn.Embedding
      qwen_linear:  nn.Module (projection)
    """

    def __init__(self, backbone, vl_input, num_frames=1, num_views=1):
        super().__init__()

        inner_model = backbone.qwen_model.model  # Qwen3VLModel
        visual = inner_model.visual
        language_model = inner_model.language_model

        input_ids = vl_input["input_ids"]
        grid_thw = vl_input["image_grid_thw"]
        device = input_ids.device

        # --- Cog-token config ---
        self.n_cog_tokens = (
            getattr(backbone, "n_cog_tokens", 0)
            if getattr(backbone, "use_cog_tokens", False)
            else 0
        )
        self.cog_mode = getattr(backbone, "cog_mode", "cog_only")

        # --- Sub-models ---
        _print("\nSetting up GraphSafeQwen3VLBackbone...")
        self.gs_visual = GraphSafeQwen3VLVisionModel(
            visual, grid_thw, num_frames=num_frames, num_views=num_views
        )

        # Determine num_views for compression (matches vanilla LayerWrapper logic)
        iwe = vl_input.get("image_wise_encoding")
        if iwe is not None:
            iwe_val = iwe.item() if isinstance(iwe, torch.Tensor) else bool(iwe)
        else:
            iwe_val = False
        compress_num_views = vl_input.get("num_views") if iwe_val else None

        self.gs_text = GraphSafeQwen3VLTextModel(
            language_model, input_ids, self.n_cog_tokens, num_views=compress_num_views
        )

        # Install into backbone so engines that still read from there find them
        inner_model.visual = self.gs_visual
        inner_model.language_model = self.gs_text

        # --- Shared modules (references, not owned) ---
        self.embed_tokens = self.gs_text._text_model.embed_tokens
        self.qwen_linear = backbone.qwen_linear
        self.image_token_id = inner_model.config.image_token_id

        # --- Static glue buffers ---
        B, L_ids = input_ids.shape
        D = self.embed_tokens.embedding_dim

        self.register_buffer("static_input_ids", input_ids.clone())

        image_mask = input_ids == self.image_token_id
        self.register_buffer("image_mask_3d", image_mask.unsqueeze(-1).expand(B, L_ids, D))

        # 3D MROPE position IDs (matches the cog-token append in the vanilla
        # backbone forward).
        # Qwen3VL uses different position IDs per axis (temporal/height/width)
        # for image tokens based on their spatial grid layout.
        L_full = L_ids + self.n_cog_tokens
        if self.n_cog_tokens > 0:
            placeholder_token_id = 248068
            meta_ids = torch.full(
                (B, self.n_cog_tokens), placeholder_token_id, dtype=input_ids.dtype, device=device
            )
            extended_input_ids = torch.cat([input_ids, meta_ids], dim=1)
        else:
            extended_input_ids = input_ids

        with torch.no_grad():
            position_ids, _ = inner_model.get_rope_index(extended_input_ids, grid_thw, None)
        # position_ids: (3, B, L_full) — proper 3D MROPE
        self.register_buffer("static_position_ids", position_ids)

        # Cog-token embedding parameter.
        if self.n_cog_tokens > 0 and hasattr(backbone, "cog_emb"):
            self.register_buffer("static_cog_emb", backbone.cog_emb.data.clone())
        else:
            self.static_cog_emb = None

        n_img = image_mask.sum().item()
        _print(
            f"  Unified backbone: B={B}, L_ids={L_ids}, D={D}, n_cog_tokens={self.n_cog_tokens}, "
            f"L_full={L_full}, image_tokens={n_img}"
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, vl_input):
        """Full VLM forward: vl_input → backbone_features.

        Args:
            vl_input: dict with at least 'pixel_values' key.
                      Other keys (input_ids, attention_mask, etc.) are ignored
                      — the model uses its own pre-computed static buffers.

        Returns:
            backbone_features: (B, M_out, D_proj) tensor
        """
        # Vision encoding
        pixel_values = vl_input["pixel_values"]
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])
        pixel_values = pixel_values.type(self.gs_visual.dtype)

        image_emb, deepstack_features = self.gs_visual(hidden_states=pixel_values, grid_thw=None)

        # Token embedding + image scatter
        dtype = self.embed_tokens.weight.dtype
        image_emb = image_emb.to(dtype=dtype)

        token_emb = self.embed_tokens(self.static_input_ids)
        token_emb = token_emb.masked_scatter(self.image_mask_3d, image_emb)

        # cog-token append
        if self.n_cog_tokens > 0 and self.static_cog_emb is not None:
            meta = self.static_cog_emb.to(dtype).unsqueeze(0).expand(token_emb.size(0), -1, -1)
            full_emb = torch.cat([token_emb, meta], dim=1)
        else:
            full_emb = token_emb

        # DeepStack — scatter vision features into full sequence for LLM
        # Build visual_pos_mask for full_emb (L_ids + cog-token)
        # image_mask_3d covers L_ids; pad with False for cog-token positions
        deepstack_add = None
        if len(deepstack_features) > 0:
            B_ds = full_emb.shape[0]
            L_full_ds = full_emb.shape[1]
            D_ds = full_emb.shape[2]
            # image_mask_3d: (B, L_ids, D) → take 2D mask, pad to L_full
            vis_mask_2d = self.image_mask_3d[:, :, 0]  # (B, L_ids) bool
            vis_mask_full = torch.cat(
                [
                    vis_mask_2d,
                    torch.zeros(B_ds, self.n_cog_tokens, dtype=torch.bool, device=full_emb.device),
                ],
                dim=1,
            )  # (B, L_full)
            vis_mask_full_3d = vis_mask_full.unsqueeze(-1).expand(B_ds, L_full_ds, D_ds)

            ds_list = []
            for ds_feat in deepstack_features:
                ds_full = torch.zeros_like(full_emb)
                ds_full = ds_full.masked_scatter(vis_mask_full_3d, ds_feat.to(dtype))
                ds_list.append(ds_full)
            deepstack_add = torch.stack(ds_list, dim=0)  # (N_ds, B, L_full, D)

        # LLM forward
        lm_out = self.gs_text(
            inputs_embeds=full_emb,
            position_ids=self.static_position_ids,
            deepstack_add=deepstack_add,
        )
        hidden_states = lm_out.last_hidden_state

        # cog-token extract — must match the vanilla backbone slice at
        # ``rldx/model/modules/backbone/adapter.py:571`` (cog_mode='cog_only').
        # Issue #29 Phase 2 renamed the string value ``meta_only`` →
        # ``cog_only``; ``patch_checkpoint.py`` migrates the saved config,
        # but this comparison was previously stale.  Without the slice
        # GraphSafeVLM and CustomVLMChain disagree on token count: the
        # custom chain slices unconditionally on ``n_cog_tokens > 0``
        # (``custom_backbone_chain.py:186``) so a stale string here makes
        # ``gs_msat.n_vl`` get baked at the unsliced 234 while the custom
        # VLM hands MSAT the sliced 64 — Path D crashes at compile time
        # and Path C silently mis-computes (cos_sim ≈ 0.96 vs vanilla).
        # Accept legacy ``meta_only`` for back-compat with any
        # checkpoint that pre-dates the rename and slipped past the
        # patch script.
        if self.n_cog_tokens > 0 and self.cog_mode in ("cog_only", "meta_only"):
            hidden_states = hidden_states[:, -self.n_cog_tokens :, :]

        # Projection
        return self.qwen_linear(hidden_states)
