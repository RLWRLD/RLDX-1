import os
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import is_torchdynamo_compiling

from rldx.model.modules.backbone.modeling_vtc import VTC_Qwen3VL
from rldx.utils.dist import rank_zero_print as _print


# Default attention implementation for the Qwen3-VL backbone load.
# Production stays on FlashAttention-2 for throughput; environments that
# cannot build flash-attn (e.g. brand-new toolchains, CI runners with no
# nvcc) can opt out via ``RLDX_ATTN_IMPL=sdpa`` without touching code.
_DEFAULT_ATTN_IMPL = os.environ.get("RLDX_ATTN_IMPL", "flash_attention_2")


class VTCQwen3VLBackbone(nn.Module):
    def __init__(
        self,
        model_name: str,
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = False,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        tune_top_llm_layers: int = 0,
        trainable_params_fp32: bool = False,
        use_cog_tokens: bool = False,
        cog_mode: str = "cog_only",
        n_cog_tokens: int = 8,
        motion_config: Optional[dict] = None,
        transformers_loading_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Args:
            tune_llm: whether to tune the LLM model (default: False)
            tune_visual: whether to tune the visual model (default: False)
            model_name: path to VTC pretrained model or base Qwen3-VL model
            motion_config: dict of motion module config (e.g., {"use_motion": True, "motion_insert_layer": 9})
            transformers_loading_kwargs: HF ``from_pretrained`` / ``snapshot_download``
                kwargs (``revision``, ``cache_dir``, ``token``). Issue #22 —
                previously swallowed by ``**kwargs`` and dropped before
                reaching the real HF calls, so ``--model-revision`` was a
                silent no-op. Now surfaced explicitly and spread into every
                weight-download / config-load site below.
        """
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here, set to False"

        # Caller's dict is the single source of truth for HF from_pretrained
        # kwargs. ``setdefault`` puts a floor under ``trust_remote_code``
        # (RLDX always needs it on) without overriding a caller-supplied
        # value, and — critically — never duplicates a key explicitly on
        # the from_pretrained call lines below. PR #43 (Issue #22) had
        # passed ``trust_remote_code=True`` *and* spread the dict, which
        # raised ``TypeError: got multiple values for keyword argument
        # 'trust_remote_code'`` for every finetune-from-HF run because
        # ``setup.py`` preloads ``trust_remote_code`` into the dict
        # (Issue #45). Keep the explicit kwarg list on each call site
        # confined to truly model-specific args (``attn_implementation``,
        # ``torch_dtype``) which ``setup.py`` does NOT put in the dict.
        transformers_loading_kwargs = dict(transformers_loading_kwargs or {})
        transformers_loading_kwargs.setdefault("trust_remote_code", True)

        skip_pretrained_weights = kwargs.pop("skip_pretrained_weights", False)
        if skip_pretrained_weights:
            _print("[i] Creating VTC-Qwen3-VL architecture only (weights from checkpoint)")
            from transformers import AutoConfig

            from rldx.model.modules.backbone.modeling_vtc import LayerWrapper

            backbone_config = AutoConfig.from_pretrained(model_name, **transformers_loading_kwargs)
            if motion_config is not None:
                for k, v in motion_config.items():
                    setattr(backbone_config.vision_config, k, v)
            backbone_config._attn_implementation = _DEFAULT_ATTN_IMPL
            self.qwen_model = VTC_Qwen3VL._from_config(backbone_config)
            for layer_idx in range(len(self.qwen_model.model.language_model.layers)):
                self.qwen_model.model.language_model.layers[layer_idx] = LayerWrapper(
                    self.qwen_model.model.language_model.layers[layer_idx],
                    layer_idx=layer_idx,
                    internal_projection=4,
                    img_pattern=[151652],
                    motion_token=1,
                )
            if load_bf16:
                self.qwen_model = self.qwen_model.to(torch.bfloat16)
        else:
            _print(f"[i] Loading VTC-Qwen3-VL model from {model_name}")
            self.qwen_model = VTC_Qwen3VL.from_pretrained(
                model_name,
                motion_config=motion_config,
                attn_implementation=_DEFAULT_ATTN_IMPL,
                torch_dtype=torch.bfloat16,
                **transformers_loading_kwargs,
            )
        # Keep BatchNorm running stats in float32 for bf16 compatibility
        motion_block = getattr(self.qwen_model.model.visual, "motion_block", None)
        if motion_block is not None:
            for m in motion_block.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                    m.float()
        self.qwen_linear = torch.nn.Identity()
        self.use_cog_tokens = bool(use_cog_tokens)
        self.cog_mode = cog_mode
        self.n_cog_tokens = int(n_cog_tokens)

        # motion module vl_input projection: vision_hidden_dim -> llm_hidden_dim
        self.motion_injection_point = (
            motion_config.get("motion_injection_point", "vision_encoder")
            if motion_config
            else "vision_encoder"
        )
        self.motion_pool_type = (
            motion_config.get("motion_pool_type", "avg") if motion_config else "avg"
        )
        self.motion_drop = motion_config.get("motion_drop", True) if motion_config else True
        self.moss_proj = None
        self.moss_spatial_conv = None
        if (
            self.motion_injection_point == "vl_input"
            and motion_config
            and motion_config.get("use_motion", False)
        ):
            vit_hidden_size = self.qwen_model.model.visual.config.hidden_size  # 1280
            llm_hidden_size = self.qwen_model.model.language_model.config.hidden_size  # 3584
            self.moss_proj = nn.Sequential(
                nn.LayerNorm(vit_hidden_size),
                nn.Linear(vit_hidden_size, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size),
            )
            # Zero-init last linear so motion module tokens start as no-ops
            nn.init.zeros_(self.moss_proj[-1].weight)
            nn.init.zeros_(self.moss_proj[-1].bias)

            if self.motion_pool_type == "conv":
                self.moss_spatial_conv = nn.Sequential(
                    nn.Conv2d(
                        vit_hidden_size,
                        vit_hidden_size,
                        kernel_size=4,
                        stride=4,
                        groups=vit_hidden_size,
                        bias=False,
                    ),
                    nn.Conv2d(vit_hidden_size, vit_hidden_size, kernel_size=1, bias=True),
                )
                _print(
                    f"[motion module vl_input] Conv pooling: depthwise separable, "
                    f"params={sum(p.numel() for p in self.moss_spatial_conv.parameters()):,}"
                )

            _print(
                f"[motion module vl_input] Created moss_proj: {vit_hidden_size} -> {llm_hidden_size}, pool_type={self.motion_pool_type}"
            )
            assert use_cog_tokens, "motion module vl_input requires use_cog_tokens=True"

        if self.use_cog_tokens:
            feature_dim = self.qwen_model.model.language_model.config.hidden_size
            self._init_cog_token_modules(feature_dim)

        total_layers = len(self.qwen_model.model.language_model.layers)

        if isinstance(select_layer, (list, tuple)):
            hs_indices = sorted({int(i) for i in select_layer})
        else:
            hs_indices = [int(select_layer)]

        for k in hs_indices:
            assert 0 <= k <= total_layers, f"select_layer {k} out of range 0..{total_layers}"

        max_k = hs_indices[-1]
        while len(self.qwen_model.model.language_model.layers) > max_k:
            self.qwen_model.model.language_model.layers.pop(-1)

        self.select_layers = hs_indices
        self._hs_idx = lambda k: k

        _print(
            f"\n[i] Select layers (hs indices): {self.select_layers} "
            f"of total_layers={total_layers}; kept {max_k} blocks"
        )

        self.set_trainable_parameters(
            tune_llm, tune_visual, tune_top_llm_layers, print_params=False
        )
        if load_bf16 and trainable_params_fp32:
            for n, p in self.named_parameters():
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)
                    _print(f"Casting trainable parameter {n} to fp32")

    def _init_cog_token_modules(self, feature_dim: int) -> None:
        if self.cog_mode not in {"full", "cog_only"}:
            raise ValueError(f"Unsupported cog_mode '{self.cog_mode}' for VTC backbone.")
        if self.n_cog_tokens <= 0:
            raise ValueError("`n_cog_tokens` must be > 0 when `use_cog_tokens` is enabled.")
        self.cog_emb = nn.Parameter(torch.randn(self.n_cog_tokens, feature_dim) * 0.02)

        _print("\n[i] cog_emb initialized in VTCQwen3VLBackbone:")
        _print(f"  Shape: {self.cog_emb.shape}")
        _print(f"  Dtype: {self.cog_emb.dtype}")
        try:
            _print(f"  Min: {self.cog_emb.min().item():.6f}")
            _print(f"  Max: {self.cog_emb.max().item():.6f}")
            _print(f"  Mean: {self.cog_emb.mean().item():.6f}")
            _print(f"  Std: {self.cog_emb.std().item():.6f}")
        except RuntimeError as e:
            if "meta tensors" in str(e):
                _print("  (Statistics not available - meta tensor)")
            else:
                raise

    def set_trainable_parameters(
        self,
        tune_llm: bool,
        tune_visual: bool,
        tune_top_llm_layers: int = 0,
        print_params: bool = True,
    ):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.qwen_model.model.language_model.requires_grad_(False)
            self.qwen_model.lm_head.requires_grad_(False)
        if not tune_visual:
            self.qwen_model.model.visual.requires_grad_(False)
            # Unfreeze motion module block even when visual is frozen
            if (
                hasattr(self.qwen_model.model.visual, "motion_block")
                and self.qwen_model.model.visual.motion_block is not None
            ):
                self.qwen_model.model.visual.motion_block.requires_grad_(True)

        if tune_top_llm_layers > 0:
            for layer in self.qwen_model.model.language_model.layers[-tune_top_llm_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        if print_params:
            _print("=" * 50)
            _print(f"[i] Tune backbone llm: {self.tune_llm}")
            _print(f"[i] Tune backbone vision tower: {self.tune_visual}")
            if tune_top_llm_layers > 0:
                _print(f"[i] Tune top {tune_top_llm_layers} LLM layers")
            _print("=" * 50 + "\n")

            trainable_params = 0
            total_params = 0
            trainable_param_names = []
            for name, p in self.named_parameters():
                total_params += p.numel()
                if p.requires_grad:
                    trainable_params += p.numel()
                    trainable_param_names.append(name)

            _print(
                f"[i] Backbone trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)"
            )

            if not tune_llm and not tune_visual:
                for name in trainable_param_names:
                    _print(f"[i] Backbone trainable parameter: {name}")
            if trainable_params == 0:
                _print("[w] No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        if self.training:
            if self.qwen_model and not self.tune_llm:
                self.qwen_model.eval()
            if self.qwen_model.model.visual and not self.tune_visual:
                self.qwen_model.model.visual.eval()
            # motion module block must stay in train mode for correct BatchNorm behavior
            motion_block = getattr(self.qwen_model.model.visual, "motion_block", None)
            if motion_block is not None:
                motion_block.train()

    def _process_moss_features(self, moss_feats, moss_meta):
        """Process saved motion module features for vl_input injection: spatial pool + projection.

        Args:
            moss_feats: (B, T, V_moss, P, D) raw motion module output at vision encoder layer
            moss_meta: (true_batch, num_frames, num_moss_views, H, W)
        Returns:
            (B, num_moss_tokens, llm_hidden_dim) pooled and projected motion module tokens
        """
        B, T, V_moss, H, W = moss_meta
        D = moss_feats.shape[-1]

        # (B, T, V, H, W, D) -> (B*V, D, T, H, W)
        moss_feats = moss_feats.reshape(B, T, V_moss, H, W, D)
        moss_feats = moss_feats.permute(0, 2, 5, 1, 3, 4).contiguous()
        moss_feats = moss_feats.reshape(B * V_moss, D, T, H, W)

        if self.moss_spatial_conv is not None:
            BV, D_, T_, H_, W_ = moss_feats.shape
            moss_feats = moss_feats.permute(0, 2, 1, 3, 4).reshape(BV * T_, D_, H_, W_)
            moss_feats = self.moss_spatial_conv(moss_feats)
            H_out, W_out = moss_feats.shape[2], moss_feats.shape[3]
            moss_feats = moss_feats.reshape(BV, T_, D_, H_out, W_out).permute(0, 2, 1, 3, 4)
        else:
            pool_h = max(1, H // 4)
            pool_w = max(1, W // 4)
            moss_feats = F.adaptive_avg_pool3d(moss_feats, (T, pool_h, pool_w))

        # Flatten to tokens: (B*V, D, T*S) -> (B, V*T*S, D)
        moss_feats = moss_feats.reshape(B * V_moss, D, -1).permute(0, 2, 1).contiguous()
        moss_feats = moss_feats.reshape(B, -1, D)

        moss_feats = self.moss_proj(moss_feats)
        return moss_feats

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def _forward_qwen_with_cog_tokens(self, qwen_input: dict):
        # Unwrap the qwen_input dictionary
        input_ids = qwen_input.get("input_ids", None)
        attention_mask = qwen_input.get("attention_mask", None)
        position_ids = qwen_input.get("position_ids", None)
        past_key_values = qwen_input.get("past_key_values", None)
        inputs_embeds = qwen_input.get("inputs_embeds", None)
        pixel_values = qwen_input.get("pixel_values", None)
        pixel_values_videos = qwen_input.get("pixel_values_videos", None)
        image_grid_thw = qwen_input.get("image_grid_thw", None)
        video_grid_thw = qwen_input.get("video_grid_thw", None)
        cache_position = qwen_input.get("cache_position", None)
        image_wise_encoding = qwen_input.get("image_wise_encoding", None)
        num_views = qwen_input.get("num_views", None)
        num_frames = qwen_input.get("num_frames", None)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.qwen_model.model.get_input_embeddings()(input_ids)

        device = inputs_embeds.device
        image_mask = None
        video_mask = None

        # Build motion module kwargs for get_image_features
        moss_kwargs = {}
        if num_frames is not None:
            moss_kwargs["num_frames"] = (
                int(num_frames[0]) if isinstance(num_frames, torch.Tensor) else int(num_frames)
            )
        if num_views is not None:
            moss_kwargs["num_views"] = (
                int(num_views[0]) if isinstance(num_views, torch.Tensor) else int(num_views)
            )

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.qwen_model.model.get_image_features(
                pixel_values, image_grid_thw, **moss_kwargs
            )
            image_embeds = torch.cat(image_embeds, dim=0).to(device, inputs_embeds.dtype)
            image_mask, _ = self.qwen_model.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.qwen_model.model.get_video_features(
                pixel_values_videos, video_grid_thw
            )
            video_embeds = torch.cat(video_embeds, dim=0).to(device, inputs_embeds.dtype)
            _, video_mask = self.qwen_model.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(
                    img_embed.device
                )
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        ### Extract and process motion module features for vl_input injection
        moss_tokens = None
        n_motion_tokens = 0
        motion_insert_pos = 0
        if self.moss_proj is not None:
            visual = self.qwen_model.model.visual
            if visual._moss_features is not None:
                moss_tokens = self._process_moss_features(
                    visual._moss_features, visual._moss_meta
                ).to(inputs_embeds.dtype)
                n_motion_tokens = moss_tokens.shape[1]
                visual._moss_features = None
                visual._moss_meta = None

        bsz = inputs_embeds.size(0)
        placeholder_token_id = 248068

        ### Insert motion module tokens BEFORE image tokens (for causal attention benefit)
        # Sequence: [text | motion module | images | trailing] + [meta_queries]
        if moss_tokens is not None:
            image_token_id = self.qwen_model.model.config.image_token_id
            first_img_pos = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0][0].item()
            motion_insert_pos = first_img_pos

            text_emb = inputs_embeds[:, :first_img_pos, :]
            image_emb = inputs_embeds[:, first_img_pos:, :]
            inputs_embeds = torch.cat([text_emb, moss_tokens, image_emb], dim=1)

            text_att = attention_mask[:, :first_img_pos]
            image_att = attention_mask[:, first_img_pos:]
            moss_att = torch.ones(bsz, n_motion_tokens, dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([text_att, moss_att, image_att], dim=1)

            text_ids = input_ids[:, :first_img_pos]
            image_ids = input_ids[:, first_img_pos:]
            moss_ids = torch.full(
                (bsz, n_motion_tokens), placeholder_token_id, dtype=input_ids.dtype, device=device
            )
            input_ids = torch.cat([text_ids, moss_ids, image_ids], dim=1)

            if visual_pos_masks is not None:
                text_vis = visual_pos_masks[:, :first_img_pos]
                image_vis = visual_pos_masks[:, first_img_pos:]
                moss_vis = torch.zeros(
                    bsz, n_motion_tokens, dtype=visual_pos_masks.dtype, device=device
                )
                visual_pos_masks = torch.cat([text_vis, moss_vis, image_vis], dim=1)

        ### Append cognition token embeddings
        meta_raw = self.cog_emb.to(inputs_embeds.dtype).unsqueeze(0).expand(bsz, -1, -1)
        full_emb = torch.cat([inputs_embeds, meta_raw], dim=1)

        ### Extend attention_mask for cognition tokens
        meta_ones = torch.ones(bsz, self.n_cog_tokens, dtype=attention_mask.dtype, device=device)
        full_att_mask = torch.cat([attention_mask, meta_ones], dim=1)

        ### Extend visual_pos_masks for cognition tokens
        if visual_pos_masks is not None:
            vis_pad = torch.zeros(
                bsz, self.n_cog_tokens, dtype=visual_pos_masks.dtype, device=device
            )
            visual_pos_masks = torch.cat([visual_pos_masks, vis_pad], dim=1)

        ### Extend input_ids with placeholder tokens
        meta_ids = torch.full(
            (bsz, self.n_cog_tokens),
            placeholder_token_id,
            dtype=input_ids.dtype,
            device=device,
        )
        extended_input_ids = torch.cat([input_ids, meta_ids], dim=1)

        # ── 3D position_ids (mirrors Qwen3VLModel.forward) ──
        if position_ids is None:
            attention_mask_tensor = (
                full_att_mask
                if not isinstance(full_att_mask, dict)
                else full_att_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = (
                        attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    )
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do assisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (extended_input_ids is not None and extended_input_ids.shape[1] != 1)
                or (full_emb is not None and full_emb.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (
                prefill_compiled_stage or prefill_noncompiled_stage
            ) or self.qwen_model.model.rope_deltas is None:
                position_ids, rope_deltas = self.qwen_model.model.get_rope_index(
                    extended_input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.qwen_model.model.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = full_emb.shape
                delta = (
                    (cache_position[0] + self.qwen_model.model.rope_deltas).to(device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # ── Language model forward ──
        # Pass motion_drop_info so LayerWrapper drops motion module tokens at internal_projection
        motion_drop_info = None
        if n_motion_tokens > 0 and self.motion_drop:
            motion_drop_info = {"start": motion_insert_pos, "count": n_motion_tokens}
        outputs = self.qwen_model.model.language_model(
            input_ids=extended_input_ids,
            position_ids=position_ids,
            attention_mask=full_att_mask,
            past_key_values=past_key_values,
            inputs_embeds=full_emb,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            output_hidden_states=True,
            image_wise_encoding=image_wise_encoding,
            num_views=num_views,
            motion_drop_info=motion_drop_info,
        )

        return outputs.last_hidden_state, full_att_mask

    def forward_qwen(self, vl_input: BatchFeature) -> BatchFeature:
        if "pixel_values" in vl_input and vl_input["pixel_values"].ndim == 3:
            pv = vl_input["pixel_values"]
            vl_input["pixel_values"] = pv.reshape(-1, pv.shape[-1])

        if "image_grid_thw" in vl_input and vl_input["image_grid_thw"].ndim == 3:
            grid = vl_input["image_grid_thw"]
            vl_input["image_grid_thw"] = grid.reshape(-1, 3)

        # Generate image_mask from input_ids
        image_mask = None
        if "input_ids" in vl_input:
            image_token_id = self.qwen_model.model.config.image_token_id
            image_mask = vl_input["input_ids"] == image_token_id  # [B, L]

        if self.use_cog_tokens:
            last_hs, attn_mask = self._forward_qwen_with_cog_tokens(vl_input)

            if self.cog_mode == "cog_only":
                features = last_hs[:, -self.n_cog_tokens :, :]
                attn_mask = attn_mask[:, -self.n_cog_tokens :]
                if image_mask is not None:
                    image_mask = image_mask[:, -self.n_cog_tokens :]
            else:
                features = last_hs

            features = self.qwen_linear(features)
            return features, attn_mask, image_mask
        else:
            qwen_output = self.qwen_model(**vl_input, output_hidden_states=True, return_dict=True)
            attn_mask = vl_input["attention_mask"]

            hs = qwen_output.hidden_states
            selected_indices = [self._hs_idx(i) for i in self.select_layers]
            feats = [hs[idx] for idx in selected_indices]

            linearized = [self.qwen_linear(feat) for feat in feats]
            features = linearized[0] if len(linearized) == 1 else linearized
            return features, attn_mask, image_mask

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        keys_to_use = [
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "image_wise_encoding",
            "num_views",
        ]
        filtered = {k: vl_input[k] for k in keys_to_use}
        if "num_frames" in vl_input:
            filtered["num_frames"] = vl_input["num_frames"]
        vl_input = filtered
        outputs, attention_mask, image_mask = self.forward_qwen(vl_input)

        return BatchFeature(
            data={
                "backbone_features": outputs,
                "backbone_attention_mask": attention_mask,
                "image_mask": image_mask,
            }
        )
