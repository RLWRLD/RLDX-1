"""Unified VLM chain: Vision Encoder + Glue + LLM Decoder.

CustomVLMChain composes CustomVisionEncoderChain and CustomLLMChain into a
single compilable nn.Module.  All static buffers (RoPE, pos_embeds, image mask,
etc.) are pre-computed by the builder and stored as registered buffers.

Forward flow:
  pixel_values → Vision Encoder → Embed + Image Scatter →
  (cog-token append) → (Pre-compression LLM) → (Context Compress) →
  (Post-compression LLM) → (cog-token extract) → Projection

Builder:
  build_custom_backbone_chain(model) → CustomVLMChain
"""

from __future__ import annotations

import os

from llm.engine.custom_llm_chain import CustomLLMChain
from llm.engine.kernels.fused_llm_attention import (
    autotune_split_m_threshold as autotune_llm_split_m_threshold,
    prepare_norm_weight_rot,
    prepare_signed_sin,
)
import torch
import torch.nn as nn
from vision_encoder.engine.custom_vision_encoder_chain import CustomVisionEncoderChain
from vision_encoder.engine.kernels.fused_vision_attention import (
    autotune_split_m_threshold as autotune_vision_split_m_threshold,
)

from rldx.utils.dist import rank_zero_print as _print


# CustomVLMChain


class CustomVLMChain(nn.Module):
    """Unified Vision + LLM chain with all static buffers pre-computed.

    Supports:
      - Basic Qwen3VL (no compression, no cog-token)
      - cog-token (append/extract learned tokens)
      - ContextVLA context compression (two-phase LLM with static indices)
    """

    def __init__(
        self,
        # Sub-chains
        vision_chain,  # CustomVisionEncoderChain
        llm_chain,  # CustomLLMChain (single-chain mode) or None
        pre_compress_chain,  # CustomLLMChain (layers 0..k-1) or None
        post_compress_chain,  # CustomLLMChain (layers k..N-1) or None
        # Vision modules
        patch_embed,  # nn.Module — visual.patch_embed
        # LLM modules
        embed_tokens,  # nn.Embedding — language_model.get_input_embeddings()
        qwen_linear,  # nn.Module — backbone.qwen_linear (projection)
        # Static vision buffers
        static_pos_embeds,  # (M_vis, D_vis) — pre-computed position embeddings
        # Static LLM buffers
        static_input_ids,  # (B, L) — input_ids for embed_tokens
        static_token_embeds,  # (B, L, D) — pre-computed text token embeddings
        image_mask_3d,  # (B, L, D) bool — image token mask for masked_scatter
        # cog-token
        n_cog_tokens=0,
        static_cog_emb=None,  # (n_cog, D) bf16 — learned cog-token embeddings
        # Context compression
        compress_begin_idx=-1,  # Python int — start of image region (-1 = no compression)
        compress_end_idx=-1,  # Python int — end of image region
    ):
        super().__init__()

        # Sub-chains
        self.vision_chain = vision_chain
        self.llm_chain = llm_chain
        self.pre_compress_chain = pre_compress_chain
        self.post_compress_chain = post_compress_chain

        # Vision / LLM modules (not owned, just referenced)
        self.patch_embed = patch_embed
        self.embed_tokens = embed_tokens
        self.qwen_linear = qwen_linear

        # Static buffers
        self.register_buffer("static_pos_embeds", static_pos_embeds)
        self.register_buffer("static_input_ids", static_input_ids)
        self.register_buffer("static_token_embeds", static_token_embeds)
        self.register_buffer("image_mask_3d", image_mask_3d)

        # cog-token
        self.n_cog_tokens = n_cog_tokens
        if static_cog_emb is not None:
            self.register_buffer("static_cog_emb", static_cog_emb)
        else:
            self.static_cog_emb = None

        # DeepStack: pre-compute visual position mask for full sequence (L_ids + cog-token)
        # image_mask_3d: (B, L_ids, D), take 2D → pad with cog-token zeros → expand to 3D
        B_m, L_ids_m, D_m = image_mask_3d.shape
        L_full_m = L_ids_m + n_cog_tokens
        vis_mask_full = torch.cat(
            [
                image_mask_3d[:, :, 0],  # (B, L_ids) bool
                torch.zeros(B_m, n_cog_tokens, dtype=torch.bool, device=image_mask_3d.device),
            ],
            dim=1,
        )  # (B, L_full)
        self.register_buffer(
            "visual_pos_mask_full_3d", vis_mask_full.unsqueeze(-1).expand(B_m, L_full_m, D_m)
        )
        self.register_buffer(
            "visual_pos_flat_indices",
            self.visual_pos_mask_full_3d.reshape(-1).nonzero(as_tuple=False).squeeze(1),
        )
        image_mask_full = torch.cat(
            [
                image_mask_3d,
                torch.zeros(B_m, n_cog_tokens, D_m, dtype=torch.bool, device=image_mask_3d.device),
            ],
            dim=1,
        )
        self.register_buffer("image_mask_full_3d", image_mask_full)
        self.register_buffer(
            "image_mask_full_flat_indices",
            image_mask_full.reshape(-1).nonzero(as_tuple=False).squeeze(1),
        )

        # Pre-bake the static full-sequence embeddings once.
        if n_cog_tokens > 0 and static_cog_emb is not None:
            meta = static_cog_emb.unsqueeze(0).expand(static_token_embeds.size(0), -1, -1)
            static_full_inputs_embeds = torch.cat([static_token_embeds, meta], dim=1)
        else:
            static_full_inputs_embeds = static_token_embeds
        self.register_buffer("static_full_inputs_embeds", static_full_inputs_embeds)
        self.register_buffer(
            "inputs_embeds_work",
            torch.empty_like(static_full_inputs_embeds),
        )

        # Context compression (trace-time constants)
        self.compress_begin_idx = compress_begin_idx
        self.compress_end_idx = compress_end_idx

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, pixel_values):
        """Full VLM forward: vision → embed → LLM → projection.

        Args:
            pixel_values: raw pixel tensor (same as backbone input)

        Returns:
            backbone_features: (B, M_out, D_proj) bf16
        """
        # Vision Encoding
        hidden = self.patch_embed(pixel_values)
        hidden = hidden + self.static_pos_embeds
        seq_len, _ = hidden.size()
        hidden = hidden.reshape(seq_len, -1)
        merged, ds_features = self.vision_chain(hidden)

        # Token Embedding + Image Scatter
        ie = self.inputs_embeds_work
        ie.copy_(self.static_full_inputs_embeds)
        ie.view(-1).index_copy_(
            0, self.image_mask_full_flat_indices, merged.to(dtype=ie.dtype).reshape(-1)
        )

        # DeepStack — pass sparse per-layer features + shared scatter indices
        deepstack_features = ds_features if len(ds_features) > 0 else None

        # LLM Decoder
        if self.compress_begin_idx < 0:
            # Simple path: single LLM chain
            out = self.llm_chain(
                ie,
                deepstack_features=deepstack_features,
                deepstack_flat_indices=self.visual_pos_flat_indices,
            )
        else:
            # VTC path: two-phase with static compression
            # DeepStack is applied in pre_compress_chain (layers 0..k-1)
            out = self.pre_compress_chain(
                ie,
                deepstack_features=deepstack_features,
                deepstack_flat_indices=self.visual_pos_flat_indices,
            )
            out = self._static_compress(out)
            out = self.post_compress_chain(out)

        # cog-token Extract (optional)
        if self.n_cog_tokens > 0:
            out = out[:, -self.n_cog_tokens :, :]

        # Projection
        return self.qwen_linear(out)

    def _static_compress(self, hidden_states):
        """Static context compression replacing VTC LayerWrapper.

        Replaces image token region [begin_idx, end_idx) with a single
        motion token (mean of dropped tokens).  Uses masked sum to match
        vanilla LayerWrapper's BF16 accumulation order.
        """
        b = self.compress_begin_idx
        e = self.compress_end_idx
        n_drop = e - b
        motion_token = hidden_states[:, b:e, :].sum(dim=1, keepdim=True) / n_drop
        keep_front = hidden_states[:, :b, :]
        keep_back = hidden_states[:, e:, :]
        return torch.cat([keep_front, motion_token, keep_back], dim=1)

    # ------------------------------------------------------------------
    # Diagnosis: forward with full intermediate capture
    # ------------------------------------------------------------------

    def forward_with_intermediates(self, pixel_values):
        """Forward with per-stage / per-layer intermediate capture.

        Returns:
            dict with keys:
                'vision_merged':         (M', D_out) — vision merger output
                'vision_intermediates':  list of dicts per vision block
                'inputs_embeds':         (B, L, D) — after scatter + cog-token
                'llm_output':            (B, M, D) — LLM normed output
                'llm_intermediates':     list of dicts per LLM layer
                'backbone_features':     (B, M_out, D_proj) — final output
        """
        refs = {}

        # Vision Encoding (with intermediates)
        hidden = self.patch_embed(pixel_values)
        hidden = hidden + self.static_pos_embeds
        seq_len, _ = hidden.size()
        hidden = hidden.reshape(seq_len, -1)
        merged, _ds_features, vis_intermediates = self.vision_chain.forward_with_intermediates(
            hidden
        )

        refs["vision_merged"] = merged.detach().clone()
        refs["vision_intermediates"] = vis_intermediates

        # Token Embedding + Image Scatter
        ie = self.embed_tokens(self.static_input_ids)
        ie = ie.masked_scatter(self.image_mask_3d, merged.to(dtype=ie.dtype))

        # cog-token Append
        if self.n_cog_tokens > 0 and self.static_cog_emb is not None:
            meta = self.static_cog_emb.unsqueeze(0).expand(ie.size(0), -1, -1)
            ie = torch.cat([ie, meta], dim=1)

        refs["inputs_embeds"] = ie.detach().clone()

        # LLM Decoder (with intermediates)
        if self.compress_begin_idx < 0:
            llm_out, llm_intermediates = self.llm_chain.forward_with_intermediates(ie)
        else:
            pre_out, pre_intermediates = self.pre_compress_chain.forward_with_intermediates(ie)
            compressed = self._static_compress(pre_out)
            post_out, post_intermediates = self.post_compress_chain.forward_with_intermediates(
                compressed
            )
            llm_out = post_out
            llm_intermediates = pre_intermediates + post_intermediates

        refs["llm_output"] = llm_out.detach().clone()
        refs["llm_intermediates"] = llm_intermediates

        # cog-token Extract
        if self.n_cog_tokens > 0:
            llm_out = llm_out[:, -self.n_cog_tokens :, :]

        # Projection
        backbone_features = self.qwen_linear(llm_out)
        refs["backbone_features"] = backbone_features.detach().clone()

        return refs


# Builder


def build_custom_backbone_chain(vlm_model, device=None, dtype=torch.bfloat16):
    """Build a CustomVLMChain from a GraphSafeQwen3VLBackbone (no compilation).

    Decomposes the model into vision / LLM components, builds a
    CustomVLMChain with baked RoPE + custom Triton ops.

    Note: RoPE cos/sin and attention accumulators use fp32 internally
    (matching PyTorch eager behavior), regardless of the model dtype.

    Args:
        vlm_model: GraphSafeQwen3VLBackbone instance
        device: torch device (default: inferred from model)
        dtype: model dtype for buffers/embeddings (default: torch.bfloat16)

    Returns:
        CustomVLMChain (uncompiled)
    """
    import backbone.llm.engine.ops  # noqa: F401 — registers vlm:: LLM ops
    import backbone.vision_encoder.engine.ops  # noqa: F401 — registers vlm:: vision ops

    if device is None:
        device = next(vlm_model.parameters()).device
    gs_visual = vlm_model.gs_visual
    gs_text = vlm_model.gs_text

    # --- Dimensions (from pre-computed buffers) ---
    # static_position_ids is (3, B, L) for 3D MROPE
    pos_ids = vlm_model.static_position_ids
    B, L = pos_ids.shape[-2], pos_ids.shape[-1]
    D = vlm_model.embed_tokens.embedding_dim

    _print(f"  [VLMChain] B={B}, L={L}, D={D}")

    # --- Vision static buffers ---
    static_pos_embeds = gs_visual.pos_embeds
    vis_pos_cos = gs_visual.pos_cos
    vis_pos_sin = gs_visual.pos_sin

    _print(
        f"  [VLMChain] Vision: pos_embeds={list(static_pos_embeds.shape)}, "
        f"cu_seqlens={list(gs_visual.cu_seqlens.shape)}, "
        f"max_seqlen={gs_visual.max_seqlen}"
    )
    enable_motion_fast = os.environ.get("CUSTOM_VLM_ENABLE_MOTION_FAST") == "1"

    # --- Vision chain ---
    vision_chain = CustomVisionEncoderChain(
        gs_visual.blocks,
        gs_visual.merger,
        gs_visual.deepstack_merger_list,
        gs_visual.deepstack_visual_indexes,
        vis_pos_cos,
        vis_pos_sin,
        gs_visual.cu_seqlens,
        gs_visual.max_seqlen,
        motion_block=gs_visual.motion_block,
        motion_insert_layer=gs_visual.motion_insert_layer,
        motion_grid_sizes=gs_visual.motion_grid_sizes,
        enable_motion_fast=enable_motion_fast,
    )
    motion_info = (
        f", Motion@layer {gs_visual.motion_insert_layer}" if gs_visual.motion_block else ""
    )
    motion_fast_info = (
        ", fast"
        if gs_visual.motion_block and enable_motion_fast
        else ", fallback"
        if gs_visual.motion_block
        else ""
    )
    _print(
        f"  [VLMChain] Vision chain: {len(gs_visual.blocks)} blocks{motion_info}{motion_fast_info}"
    )

    # --- LLM RoPE (from static_position_ids, no forward pass) ---
    # static_position_ids is already (3, B, L) for 3D MROPE
    position_ids = vlm_model.static_position_ids  # (3, B, L) MROPE
    dummy_embeds = torch.empty(B, L, D, device=device, dtype=dtype)

    with torch.no_grad():
        pos_cos, pos_sin = gs_text.rotary_emb(dummy_embeds, position_ids)
    if pos_cos.dim() == 3:
        pos_cos = pos_cos.squeeze(0)
    if pos_sin.dim() == 3:
        pos_sin = pos_sin.squeeze(0)
    pos_cos = pos_cos.contiguous()
    pos_sin = pos_sin.contiguous()

    # --- Compression info ---
    decoder_layers = list(gs_text.layers)
    num_llm_layers = len(decoder_layers)
    compress_layer_idx = -1
    compress_begin_idx = -1
    compress_end_idx = -1

    ci = gs_text.compress_info
    if ci is not None:
        compress_layer_idx = ci["compress_layer_idx"]
        compress_begin_idx = ci["static_begin"]
        compress_end_idx = ci["static_end"]
        _print(
            f"  [VLMChain] VTC compression at layer {compress_layer_idx}: "
            f"begin={compress_begin_idx}, end={compress_end_idx}"
        )

    # --- LLM chain(s) ---
    norm = gs_text.norm

    # Get head_dim from first decoder layer
    first_raw = decoder_layers[0]
    if hasattr(first_raw, "layer") and hasattr(first_raw, "internal_projection"):
        first_raw = first_raw.layer
    head_dim = first_raw.self_attn.head_dim
    num_heads = first_raw.self_attn.q_proj.weight.shape[0] // head_dim
    num_kv_heads = first_raw.self_attn.k_proj.weight.shape[0] // head_dim

    # --- Dispatch policy autotune (build-time, fixed-shape safe) ---
    # Read vision dims from gs_visual (distinct from LLM's num_heads/head_dim).
    with torch.no_grad():
        vis_attn = gs_visual.blocks[0].attn
        vis_qkv = torch.randn(
            (static_pos_embeds.shape[0], 3 * vis_attn.num_heads * vis_attn.head_dim),
            device=device,
            dtype=dtype,
        )
        best_vision_threshold = autotune_vision_split_m_threshold(
            [
                (
                    vis_qkv,
                    vis_pos_cos,
                    vision_chain.rope_sin,
                    gs_visual.cu_seqlens,
                    vis_attn.scaling,
                    vis_attn.num_heads,
                    vis_attn.head_dim,
                )
            ]
        )
        _print(
            f"  [VLMChain] Vision split threshold autotuned to {best_vision_threshold} "
            f"(vision: heads={vis_attn.num_heads}, head_dim={vis_attn.head_dim})"
        )

    # Pre-compute signed_sin (sign pattern from rotate_half folded in)
    with torch.no_grad():
        pre_signed_sin = prepare_signed_sin(pos_sin, head_dim)
        llm_workloads = []
        llm_qkv_dim = num_heads * head_dim + 2 * num_kv_heads * head_dim
        llm_workloads.append(
            (
                torch.randn((pos_cos.shape[0], llm_qkv_dim), device=device, dtype=dtype),
                first_raw.self_attn.q_norm.weight.data,
                prepare_norm_weight_rot(first_raw.self_attn.q_norm.weight.data, head_dim),
                first_raw.self_attn.k_norm.weight.data,
                prepare_norm_weight_rot(first_raw.self_attn.k_norm.weight.data, head_dim),
                pos_cos,
                pre_signed_sin,
                num_heads,
                num_kv_heads,
                head_dim,
            )
        )

    if compress_layer_idx < 0:
        llm_chain = CustomLLMChain(decoder_layers, norm, pos_cos, pre_signed_sin)
        pre_compress_chain = None
        post_compress_chain = None
        _print(f"  [VLMChain] LLM chain: {num_llm_layers} layers (single)")
    else:
        pre_layers = decoder_layers[:compress_layer_idx]
        post_layers = decoder_layers[compress_layer_idx:]

        pre_compress_chain = CustomLLMChain(
            pre_layers,
            norm,
            pos_cos,
            pre_signed_sin,
            apply_final_norm=False,
        )

        with torch.no_grad():
            post_cos = torch.cat(
                [
                    pos_cos[:compress_begin_idx],
                    pos_cos[compress_begin_idx : compress_begin_idx + 1],
                    pos_cos[compress_end_idx:],
                ],
                dim=0,
            ).contiguous()
            post_sin = torch.cat(
                [
                    pos_sin[:compress_begin_idx],
                    pos_sin[compress_begin_idx : compress_begin_idx + 1],
                    pos_sin[compress_end_idx:],
                ],
                dim=0,
            ).contiguous()
            post_signed_sin = prepare_signed_sin(post_sin, head_dim)
            post_first_raw = post_layers[0]
            if hasattr(post_first_raw, "layer") and hasattr(post_first_raw, "internal_projection"):
                post_first_raw = post_first_raw.layer
            llm_workloads.append(
                (
                    torch.randn((post_cos.shape[0], llm_qkv_dim), device=device, dtype=dtype),
                    post_first_raw.self_attn.q_norm.weight.data,
                    prepare_norm_weight_rot(post_first_raw.self_attn.q_norm.weight.data, head_dim),
                    post_first_raw.self_attn.k_norm.weight.data,
                    prepare_norm_weight_rot(post_first_raw.self_attn.k_norm.weight.data, head_dim),
                    post_cos,
                    post_signed_sin,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                )
            )

        best_llm_threshold = autotune_llm_split_m_threshold(llm_workloads)
        _print(f"  [VLMChain] LLM split threshold autotuned to {best_llm_threshold}")

        post_compress_chain = CustomLLMChain(post_layers, norm, post_cos, post_signed_sin)

        llm_chain = None
        _print(
            f"  [VLMChain] LLM chain: {len(pre_layers)} pre + {len(post_layers)} post "
            f"(compression at layer {compress_layer_idx})"
        )
    if compress_layer_idx < 0:
        best_llm_threshold = autotune_llm_split_m_threshold(llm_workloads)
        _print(f"  [VLMChain] LLM split threshold autotuned to {best_llm_threshold}")

    # --- cog-token ---
    n_cog_tokens = vlm_model.n_cog_tokens
    static_cog_emb = None

    if n_cog_tokens > 0 and vlm_model.static_cog_emb is not None:
        static_cog_emb = vlm_model.static_cog_emb.to(dtype=dtype).clone()
        _print(f"  [VLMChain] cog-token: n={n_cog_tokens}, shape={list(static_cog_emb.shape)}")
    else:
        _print("  [VLMChain] cog-token: disabled")

    with torch.no_grad():
        static_token_embeds = vlm_model.embed_tokens(vlm_model.static_input_ids).to(dtype=dtype)

    # --- Assemble ---
    backbone_chain = CustomVLMChain(
        vision_chain=vision_chain,
        llm_chain=llm_chain,
        pre_compress_chain=pre_compress_chain,
        post_compress_chain=post_compress_chain,
        patch_embed=gs_visual.patch_embed,
        embed_tokens=vlm_model.embed_tokens,
        qwen_linear=vlm_model.qwen_linear,
        static_pos_embeds=static_pos_embeds,
        static_input_ids=vlm_model.static_input_ids,
        static_token_embeds=static_token_embeds,
        image_mask_3d=vlm_model.image_mask_3d,
        n_cog_tokens=n_cog_tokens,
        static_cog_emb=static_cog_emb,
        compress_begin_idx=compress_begin_idx,
        compress_end_idx=compress_end_idx,
    )

    _print("  [VLMChain] Built successfully")

    return backbone_chain


def compile_custom_backbone_chain(
    backbone_chain, sample_input, compile_mode="max-autotune", fullgraph=True
):
    """Compile a CustomVLMChain with torch.compile and trigger compilation.

    Args:
        backbone_chain: CustomVLMChain instance (from build_custom_backbone_chain)
        sample_input: sample pixel_values tensor (triggers compilation)
        compile_mode: torch.compile mode
        fullgraph: when True (default) torch.compile errors on any
            graph break, forcing the whole forward into one FX graph.

    Returns:
        (compiled_forward, compile_time_s)
    """
    import time as _time

    _print(f"  [VLMChain] Compiling ({compile_mode}, fullgraph={fullgraph})...")
    compiled_forward = torch.compile(backbone_chain.forward, mode=compile_mode, fullgraph=fullgraph)

    t0 = _time.time()
    with torch.no_grad():
        compiled_forward(sample_input)
    torch.cuda.synchronize()
    compile_time_s = _time.time() - t0
    _print(f"  [VLMChain] Compilation: {compile_time_s:.1f}s")

    return compiled_forward, compile_time_s
