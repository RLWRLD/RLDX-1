# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified from the original NVIDIA Isaac GR00T v1.7.
# Original source: https://github.com/NVIDIA/Isaac-GR00T

from dataclasses import MISSING, asdict, dataclass, field, is_dataclass
from enum import Enum
import json
from pathlib import Path

import torch
from transformers import PretrainedConfig

from . import register_model_config


@dataclass
class RLDXConfig(PretrainedConfig):
    """Unified configuration for RLDX model with backbone and action model."""

    # Model identification
    model_type: str = "RLDX-1"
    model_dtype: str = "bfloat16"  # Use bfloat16 for Flash Attention compatibility

    # Backbone configuration
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    # Only "vtc_qwen3_vl" is supported after Issue #36 removed the vanilla
    # Qwen3 backbone path. Field is retained so patch_checkpoint /
    # compare_model_configs can still key off it, but any other value raises
    # at model-construction time (BACKBONE_REGISTRY is gone).
    backbone_model_type: str = "vtc_qwen3_vl"
    model_revision: str | None = None
    tune_top_llm_layers: int = 0  # Number of top LLM layers to tune
    backbone_embedding_dim: int = 4096  # project_to_dim
    tune_llm: bool = False
    tune_visual: bool = False
    select_layer: int = 18
    reproject_vision: bool = False
    use_flash_attention: bool = True
    load_bf16: bool = True  # Enable BF16 loading
    backbone_trainable_params_fp32: bool = True
    freeze_cog_tokens: bool = False  # Freeze cog_emb to prevent VLM backprop

    ### Image pipeline parameters
    # Step 1 — aspect-ratio-preserving resize + m-aligned crop
    image_max_area: int = 65536  # 256 * 256 by default
    image_resize_m: int = 32
    # Step 2 — optional fractional crop + resize-back (train: random, eval: center)
    random_crop_fraction: float | None = None  # None = no-op
    # Step 3 — optional photometric / geometric augmentation (train only)
    random_rotation_angle: int | None = None
    color_jitter_params: dict[str, float] | None = None
    formalize_language: bool = True
    apply_sincos_state_encoding: bool = (
        False  # Global flag to enable per-embodiment sin/cos encoding
    )
    use_relative_action: bool = True
    use_percentiles: bool = True
    conversation_image_first: bool = False

    # Action head configuration parameters
    max_state_dim: int = 64  # Default from state_shape
    max_action_dim: int = 64  # Default from action_shape
    action_horizon: int = 16
    hidden_size: int = 1024
    input_embedding_dim: int = 1536
    general_embodiment_train_ratio: float = 0

    # Global parameters from YAML
    add_pos_embed: bool = True
    attn_dropout: float = 0.2
    use_vlln: bool = False
    max_seq_len: int = 1024

    # MSAT configuration
    n_cog_tokens: int = 64
    diffusion_model_cfg: dict = field(
        default_factory=lambda: {
            "attention_head_dim": 64,
            "depth_multi_stream": 4,
            "depth_single_stream": 8,
            "dropout": 0.2,
            "final_dropout": True,
            "num_attention_heads": 24,
            "output_dim": 1024,
            "positional_embeddings": "rope_sa_only",
            "rope_theta": 10000.0,
            "temb_type": "input_token",
            "use_swiglu": True,
            "action_model_max_seq_len": 512,
            "pre_norm": "layer_norm",
            "qk_norm": "rms_norm",
        }
    )

    # Flow matching parameters
    num_inference_timesteps: int = 4
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000

    # Training parameters
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True

    # Action model (MSAT) LoRA — Issue #54. ``action_model_*`` mirrors the
    # post-Issue#29 cutover (``action_head`` → ``action_model``).
    # When ``action_model_use_lora=True``, ``RLDXActionModel.set_trainable_parameters``
    # injects PEFT LoRA adapters into the MSAT linear projections listed in
    # ``action_model_lora_target_modules`` instead of full-tuning the DiT. The
    # default target list covers MSAT's V-L / state-action / physics QKV +
    # output projections + the MMDiT inner FFN linears (see
    # ``rldx/model/modules/action_model/blocks.py`` lines 124, 320, 352, 690,
    # 1158, 1187, 1216). Targets that don't exist in the current MSAT (e.g.
    # ``p_qkv``/``p_proj`` when ``use_physics=False``) are filtered before the
    # PEFT call.
    action_model_use_lora: bool = False
    action_model_lora_rank: int = 16
    action_model_lora_alpha: int = 32
    action_model_lora_dropout: float = 0.0
    action_model_lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "vl_qkv",
            "vl_proj",
            "sa_qkv",
            "sa_proj",
            "p_qkv",
            "p_proj",
            "linear1",
            "linear2",
        ]
    )

    # Backbone (Qwen3 LLM) LoRA — Issue #55. Mirror of the action-model
    # surface: ``backbone_use_lora`` toggles PEFT injection into the LLM
    # layers; ``backbone_lora_num_layers`` picks the top-N suffix
    # (-1 = all layers, 0 = skip, N > 0 = last N). When LoRA is active the
    # backbone falls back to ``requires_grad_(False)`` first and only the
    # injected LoRA params remain trainable — so ``tune_top_llm_layers``
    # is effectively ignored (the launcher warns about the conflict).
    # ``backbone_lora_target_modules`` covers Qwen3 attention + MLP
    # projections.
    backbone_use_lora: bool = False
    backbone_lora_rank: int = 16
    backbone_lora_alpha: int = 32
    backbone_lora_dropout: float = 0.0
    backbone_lora_num_layers: int = -1
    backbone_lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # State Augmentation parameters
    state_dropout_prob: float = 0.0  # State dropout probability
    state_additive_noise_scale: float = 0.0  # Scale for additive Gaussian noise on state features

    # Real-Time Chunking (RTC). See rldx/model/modules/action_model/rtc.py.
    #   training_max_delay: if > 0, each training step samples a per-sample
    #     prefix length in [0, max_delay]; those positions use ground-truth
    #     clean actions and do not contribute to the loss.
    #   inference_mode: "none" / "trained" / "guided". Any non-none
    #     mode requires a non-zero inference_delay and an action_prefix in
    #     action_input at inference time.
    # NOTE: rtc_jacobian_beta is intentionally NOT a model-config field — it
    # is an inference-time guidance knob, not a property of the trained model.
    # The eval server (`run_rldx_server.py`) accepts it as a CLI override and
    # `rtc_config_from_rldx` falls back to RTCConfig's default when absent.
    rtc_training_max_delay: int = 0
    rtc_inference_mode: str = "none"
    rtc_inference_delay: int = 0
    rtc_inference_exec_horizon: int = 0  # s; 0 => action_horizon - inference_delay
    # Apply Jacobian guidance only on the first N denoising steps. Default 3
    # (skip last τ→1 step where VJP residual is mostly numerical noise on
    # RLDX/MSAT). None = all steps, 1 = cheapest single-step variant.
    rtc_jacobian_steps_only: int | None = 3

    # Memory configuration
    use_memory: bool = False  # Enable memory-augmented cognition tokens
    memory_length: int = 4  # Number of past timesteps for memory (= context_window)
    memory_n_cog_tokens: int | None = (
        None  # Number of cognition tokens routed through memory (defaults to n_cog_tokens)
    )
    concat_memory: bool = (
        False  # If True, concat MQ_augmented after MQ_original instead of replacing
    )
    memory_dropout_prob: float = (
        0.0  # Dropout ratio for augmented cognition tokens (concat_memory=True only, mask-out)
    )
    memory_stride: int = (
        16  # Action-step stride between memory snapshots (should match execution_horizon)
    )
    memory_cfg: dict = field(
        default_factory=lambda: {
            "hidden_size": 4096,
            "intermediate_size": 16384,
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "max_position_embeddings": 32,
            "rms_norm_eps": 1e-5,
            "use_causal_attn": True,
            "use_rope": True,
        }
    )

    # motion module configuration
    use_motion: bool = False
    motion_insert_layer: int = 9
    motion_d_hid: int = 512
    motion_window: tuple = (5, 9, 9)
    motion_ext_chnls: tuple = (256,)
    motion_int_chnls: tuple = (256, 256, 512)
    motion_corr_func: str = "cosine"
    motion_n_encoders: int = 1
    motion_use_layerscale: bool = False
    motion_layerscale_init: float = 1e-5
    motion_use_layernorm: bool = False
    motion_use_syncbn: bool = False
    motion_injection_point: str = "vision_encoder"  # "vision_encoder" or "vl_input"
    motion_pool_type: str = "avg"  # "avg" or "conv" (spatial pooling for vl_input)
    motion_drop: bool = True  # drop motion module tokens at internal_projection layer
    motion_gradient_check: bool = False  # log motion module gradient norms during training
    motion_int_mode: str = (
        "lite"  # "lite" (1x1 Conv3d L-fuse, default) or "full" (3-layer 3x3 conv stack)
    )

    # Video input configuration
    # ``use_video`` is an architectural invariant in the release codebase
    # (Issue #53): every supported checkpoint embeds VTC video tokens. The
    # field is kept on ``RLDXConfig`` so it survives ``save_pretrained`` /
    # ``from_pretrained`` round-trips (legacy checkpoints carrying the
    # field are still readable), but it is no longer a CLI knob — see
    # ``rldx/experiment/features/video.py``.
    use_video: bool = True
    video_length: int = 4
    video_stride: int = 2  # Action-step stride between video frames in context window

    # Physics (tactile/torque) configuration
    use_physics: bool = False
    physics_keys: list[str] = field(default_factory=list)  # e.g., ["tactile", "torque"]
    physics_dims: list[int] = field(
        default_factory=list
    )  # Per-key dimensions, aligned with `physics_keys` (e.g., [30, 7])
    physics_loss_weight: float = 0.1
    allow_missing_physics: bool = (
        False  # If True, samples without physics data are zero-filled and attention-masked
    )
    physics_delta_indices: list[int] | None = (
        None  # Injected from modality_configs in setup.py; d<=0 = hist, d>0 = fut
    )
    physics_use_flow_matching: bool = (
        True  # False to force legacy mode (all conditioning + MSE loss)
    )
    physics_dropout_prob: float = 0.0
    """Per-sample dropout on physics conditioning tokens during training.
    Flow-matching mode drops only history tokens; legacy mode drops the full sequence."""

    @property
    def physics_dim(self) -> int:
        """Total physics dimension, derived from physics_dims."""
        return sum(self.physics_dims) if self.physics_dims else 0

    # Multi-embodiment parameters
    max_num_embodiments: int = 36

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Ensures that all dataclass defaults (including those using default_factory)
        # are explicitly assigned to the instance, even if dataclasses initialization or subclassing
        # (PretrainedConfig) interferes with normal default injection.
        self._fill_missing_defaults()

    def _fill_missing_defaults(self):
        """Set default values for any dataclass fields not yet on the instance."""
        for f in self.__dataclass_fields__.values():
            if not hasattr(self, f.name):
                if f.default is not MISSING:
                    setattr(self, f.name, f.default)
                elif getattr(f, "default_factory", MISSING) is not MISSING:
                    setattr(self, f.name, f.default_factory())

    def __getattr__(self, name: str):
        # Strict: no silent default fallback. Every declared dataclass field
        # is populated by `_fill_missing_defaults()` in __init__, so reaching
        # __getattr__ means the attribute was never declared on this class
        # — which usually indicates a config.json written by an older version
        # of the code (e.g. `n_meta_queries` after that field was renamed to
        # `n_cog_tokens`). Previously this path silently injected the new
        # field's default and hid the migration gap; after Issue #29 the
        # only supported migration path is `scripts/patch_checkpoint.py`, so
        # we fail fast here.
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def to_filtered_dict(self, exclude_augment: bool = True) -> dict:
        """Return a dictionary representation of this config, optionally excluding augmentation keys."""
        if is_dataclass(self):
            cfg = asdict(self)
        else:
            cfg = dict(self.__dict__)

        if exclude_augment:
            exclude_keys = {
                "random_rotation_angle",
                "color_jitter_params",
                "random_crop_fraction",
                "formalize_language",
            }
            cfg = {k: v for k, v in cfg.items() if k not in exclude_keys}

        return cfg

    def to_filtered_json(self, exclude_augment: bool = True, **kwargs) -> str:
        """Return a JSON string of this config, optionally excluding augmentation keys."""

        def default(o):
            if isinstance(o, (Path, torch.dtype, torch.device)):
                return str(o)
            if isinstance(o, Enum):
                return o.value
            return str(o)

        return json.dumps(
            self.to_filtered_dict(exclude_augment),
            indent=2,
            default=default,
            **kwargs,
        )


register_model_config("RLDX-1", RLDXConfig)
