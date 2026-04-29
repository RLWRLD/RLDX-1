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

from dataclasses import dataclass

from rldx.data.embodiment_tags import EmbodimentTag


@dataclass
class TrainConfig:
    """
    Unified configuration for pretraining and fine-tuning a Vision-Language-Action (VLA) model.

    - Finetune mode: use dataset_path (or dataset_paths + dataset_mix_ratios) and embodiment_tag.
    - Pretrain mode: use pt_dataset_root and pt_dataset_mix.

    These dataset configs are mutually exclusive; do not mix finetune and pretrain dataset args.
    """

    # Model configuration --------------------------------------------------------------------------

    backbone_path: str | None = None
    """Path to backbone model (e.g., 'Qwen/Qwen3-VL-8B-Instruct' or a VTC checkpoint).
    If None, defaults to 'Qwen/Qwen3-VL-8B-Instruct' for from-scratch training,
    or uses the checkpoint's saved value."""

    base_model_path: str | None = None
    """Path to the pretrained base model checkpoint (e.g., HuggingFace Hub or local directory)."""

    model_revision: str | None = None
    """Git commit / branch / tag to pin when ``--base-model-path`` is a HF Hub
    repo id. None = HEAD (reproducibility hazard for a moving branch). Threads
    through ``snapshot_download`` in ``launch_train`` and every nested
    ``from_pretrained`` call inside the VTC backbone via
    ``transformers_loading_kwargs['revision']``."""

    backbone_select_layer: int | None = None
    """Override ``RLDXConfig.select_layer`` — the LLM layer below which the
    backbone is truncated (layers above this index are dropped at build
    time). None = keep the checkpoint-loaded / RLDXConfig default (18).
    Mostly used for layer-ablation experiments."""

    n_cog_tokens: int = 64
    """Number of meta query tokens to use (for MSATv1)."""

    action_horizon: int = 16
    """Number of future action steps predicted per chunk. Must equal the length
    of every action modality's delta_indices in the modality_config — the
    dataloader and MSAT both rely on this width, so a mismatch is caught at
    assembly time rather than during forward."""

    # ----------------------------------------------------------------------------------------------

    # Dataset config -------------------------------------------------------------------------------
    dataset_path: str | None = None
    """Path to the dataset root directory for fine-tuning (single dataset)."""

    dataset_paths: list[str] | None = None
    """
    Optional list of dataset root directories for multi-dataset fine-tuning.
    If provided, this takes precedence over dataset_path.
    Mutually exclusive with pt_dataset_root and pt_dataset_mix.
    """

    dataset_mix_ratios: list[float] | None = None
    """
    Optional per-dataset mix ratios used with dataset_paths.
    If None, all datasets are assigned a ratio of 1.0.
    """

    embodiment_tag: EmbodimentTag | None = None
    """Identifier specifying which embodiment (robot configuration) this fine-tuning run targets.
    Required for finetune mode (dataset_path / dataset_paths) — the launcher
    fails fast with ValueError if this is None. Pretrain mode (pt_dataset_root /
    pt_dataset_mix) pulls per-dataset tags from the mix definition and ignores
    this field."""

    modality_config_path: str | None = None
    """
    Path to a Python file defining the modality configuration for the given embodiment.
    If None, use the pre-registered modality config in rldx/configs/data/embodiment_configs.py.
    """

    conversation_image_first: bool = False
    """If True, use image first in conversation"""
    # ----------------------------------------------------------------------------------------------

    # Pretrain-only dataset config ('pt_' prefix) --------------------------------------------------
    pt_dataset_root: str | None = None
    """Path to the dataset root directory for pre-training.
    Used with pt_dataset_mix. Mutually exclusive with dataset_path and dataset_paths."""

    pt_dataset_mix: str | None = None
    """Dataset mix name for pre-training (see rldx/configs/data/dataset_mix.py).
    Used with pt_dataset_root. Mutually exclusive with dataset_path and dataset_paths."""

    general_embodiment_train_ratio: float = 0
    """Ratio to train general_embodiment encoder/decoder"""
    # ----------------------------------------------------------------------------------------------

    # Model Tuning Flags ---------------------------------------------------------------------------
    tune_llm: bool = False
    """If True, fine-tune the language model (LLM) backbone during training."""

    tune_visual: bool = False
    """If True, fine-tune the visual encoder (e.g., ViT or CNN backbone)."""

    tune_projector: bool = True
    """If True, fine-tune the multimodal projector layers that map vision/language features to a shared space."""

    tune_diffusion_model: bool = True
    """If True, fine-tune the diffusion-based action decoder (if present in the model).
    Ignored when ``action_model_use_lora=True`` (LoRA controls the diffusion model
    instead — see Issue #54)."""

    # Action model (MSAT) LoRA — Issue #54.
    # Naming uses ``action_model_*`` to match the post-Issue#29 cutover from
    # ``action_head`` → ``action_model``. The trainable surface is the
    # diffusion DiT inside ``RLDXActionModel.model``.
    action_model_use_lora: bool = False
    """If True, apply LoRA adapters to the action model (MSAT) instead of full
    fine-tuning. Overrides ``tune_diffusion_model`` (LoRA adapters control the
    MSAT trainable state)."""

    action_model_lora_rank: int = 16
    """LoRA rank (r) for the action model."""

    action_model_lora_alpha: int = 32
    """LoRA alpha for the action model."""

    action_model_lora_dropout: float = 0.0
    """LoRA dropout for the action model."""

    # Backbone (Qwen3 LLM) LoRA — Issue #55. Stacked on top of Issue #54's
    # action-model LoRA: same PEFT plumbing, but the adapter target is the
    # backbone LLM layers. ``backbone_lora_num_layers`` selects a top-N suffix
    # of LLM layers (-1 = all, 0 = disable, N > 0 = last N). When LoRA is on
    # ``tune_top_llm_layers`` is ignored — LoRA owns the LLM trainable
    # surface — and a launcher warning is printed if both are set.
    backbone_use_lora: bool = False
    """If True, apply LoRA adapters to the backbone LLM layers instead of full
    fine-tuning. When enabled, tune_top_llm_layers is effectively ignored
    (LoRA controls which LLM layers are adapted via backbone_lora_num_layers)."""

    backbone_lora_rank: int = 16
    """LoRA rank (r) for the backbone LLM."""

    backbone_lora_alpha: int = 32
    """LoRA alpha for the backbone LLM."""

    backbone_lora_dropout: float = 0.0
    """LoRA dropout for the backbone LLM."""

    backbone_lora_num_layers: int = -1
    """How many top LLM layers to wrap with LoRA when backbone_use_lora=True.
    -1 = all layers, 0 = disabled, N > 0 = top-N layers only."""

    tune_top_llm_layers: int = 4
    """Number of top LLM layers to tune."""

    state_dropout_prob: float = 0.0
    """Dropout probability applied to state inputs for regularization during training."""

    # ── Real-Time Chunking (RTC, arXiv 2506.07339 + 2512.05964) ─────────────────
    rtc_training_max_delay: int = 0
    """Training-time RTC: for each sample, draw prefix length d ~ U[0, max_delay]
    and condition on the clean prefix while masking its loss. 0 disables."""

    rtc_inference_mode: str = "none"
    """Inference-time RTC mode: 'none' | 'trained' (requires a checkpoint trained
    with rtc_training_max_delay > 0) | 'guided' (Jacobian universal guidance,
    works on any flow-matching checkpoint)."""

    rtc_inference_delay: int = 0
    """Frozen prefix length d carried over from previous chunk at inference time."""

    rtc_inference_exec_horizon: int = 0
    """Execution horizon s (actions consumed before replanning). 0 => action_horizon - d."""

    # NOTE: rtc_jacobian_beta is intentionally NOT exposed at the training CLI:
    # it is a pure inference-time guidance knob with no effect on the trained
    # weights. Use `--rtc-jacobian-beta` on `run_rldx_server.py` to override
    # at deployment time; otherwise RTCConfig's default (5.0) is used.
    rtc_jacobian_steps_only: int | None = 3
    """Apply Jacobian guidance only on the first N denoising steps. Default 3
    (skip last τ→1 step where VJP residual is mostly numerical noise).
    None = all steps, 1 = cheapest single-step variant."""
    # ────────────────────────────────────────────────────────────────────────────

    freeze_cog_tokens: bool = False
    """If True, freeze the cog_emb parameter in the backbone to prevent VLM layer
    backpropagation."""
    # ----------------------------------------------------------------------------------------------

    # Video Module (cf. VTC) ----------------------------------------------------------------
    # NOTE: there is intentionally no ``use_video`` CLI knob — the release
    # codebase always builds the VTC backbone (Issue #36 / PR #39 dropped the
    # vanilla Qwen3 path) and every released checkpoint is trained with video.
    # ``video_length`` / ``video_stride`` are the only video-shape parameters
    # left as CLI overrides.
    video_length: int = 4
    """Number of video frames to use as input."""

    video_stride: int = 2
    """Stride (in action step units) between consecutive video frames in the
    context window. Used to compute video frame deltas:
    [(i - (L-1)) * stride for i in range(L)]."""
    # ----------------------------------------------------------------------------------------------

    # Motion Module (cf. motion module) ---------------------------------------------------------------------
    use_motion: bool = False
    """If True, enable motion module (Motion-Aware Spatio-Temporal Summarization) block in vision encoder."""

    motion_insert_layer: int = 9
    """Layer index to insert motion module block in vision encoder."""

    motion_injection_point: str = "vision_encoder"
    """Where to inject motion module features: 'vision_encoder' (residual add at insert layer)
    or 'vl_input' (project and prepend as LLM input tokens)."""

    motion_pool_type: str = "avg"
    """Spatial pooling for vl_input motion module tokens: 'avg' (adaptive avg pool) or 'conv' (learned depthwise separable conv)."""

    motion_drop: bool = True
    """If True, drop motion module tokens at internal_projection layer (layer 4).
    If False, keep motion module tokens through all LLM layers (~40% more compute)."""

    motion_gradient_check: bool = False
    """If True, register backward hooks on motion module output to log gradient norms during training."""
    # ----------------------------------------------------------------------------------------------

    # Memory Module (cf. HAMLET) --------------------------------------------------------------------
    use_memory: bool = False
    """If True, enable memory-augmented cognition tokens for temporal context aggregation."""

    memory_length: int = 4
    """Number of past timesteps for memory context window (default: 4, using STRIDE=16)."""

    memory_n_cog_tokens: int | None = None
    """Number of cognition tokens to pass through the memory module.
    Must be <= n_cog_tokens. If None, defaults to n_cog_tokens (all cognition tokens go through memory)."""

    concat_memory: bool = False
    """If True, concatenate memory-augmented tokens after the original cognition tokens instead of
    replacing the memory subset."""

    blockwise_attn_for_memory: bool = False
    """If True, use block-wise attention (use_causal_attn=False) in the memory module
    instead of standard token-level causal attention. Block-wise attention allows
    bidirectional attention within each timestep's cognition tokens while maintaining
    causal ordering across timesteps."""

    memory_dropout_prob: float = 0.0
    """Dropout probability applied to memory-augmented cognition tokens during training.
    Only applicable when concat_memory=True. With this probability, the augmented cognition
    tokens are masked out from attention, forcing the model to predict actions from original cognition tokens only."""

    memory_stride: int = 16
    """Stride (in action step units) between consecutive memory snapshots.
    Should equal the intended `execution_horizon` at inference time so that
    memory slot boundaries align with action-chunk executions.
    Used to compute memory anchor indices: [-(L-1-i) * stride for i in range(L)]."""
    # ----------------------------------------------------------------------------------------------

    # Physics Module (tactile/torque) --------------------------------------------------------------
    use_physics: bool = False
    """If True, enable physics (tactile/torque) conditioning stream in the action head."""

    physics_keys: list[str] | None = None
    """List of physics signal keys to use, e.g. ["tactile", "torque"].
    The signals are concatenated into a single physics vector of dimension sum(physics_dims)."""

    physics_dims: list[int] | None = None
    """Per-key dimensions aligned with physics_keys, e.g. [30, 7].
    len(physics_dims) must equal len(physics_keys). Total physics_dim = sum(physics_dims)."""

    physics_loss_weight: float = 0.1
    """Weight for the physics signal prediction loss."""

    allow_missing_physics: bool = False
    """If True, datasets without physics_keys are allowed when use_physics=True.
    Samples without physics data get zero-filled physics tensors with attention mask=0."""

    physics_dropout_prob: float = 0.0
    """Per-sample dropout probability applied to physics conditioning tokens during training.
    Replaces the conditioning token sequence with a learned mask token. In flow-matching
    mode, only the history tokens are dropped (future tokens are the prediction target).
    Training-only — has no effect at eval/inference."""
    # ----------------------------------------------------------------------------------------------

    # Image pipeline -------------------------------------------------------------------------------
    image_max_area: int = 65536
    """Area budget (in pixels) for the aspect-ratio-preserving resize step.
    Defaults to 256*256 = 65536. For a 480x640 input this produces 192x256; for
    a 256x256 input it is a no-op."""

    image_resize_m: int = 32
    """Alignment multiple for both output dimensions after the aspect-area resize.
    Defaults to 32."""

    random_crop_fraction: float | None = None
    """Optional fractional crop applied after the aspect-area resize.
    Training uses a random crop position; evaluation uses a center crop.
    The cropped region is resized back to the pre-crop shape so downstream
    stages always see a fixed output size. None = no-op (default)."""
    # ----------------------------------------------------------------------------------------------

    # Data Augmentation ----------------------------------------------------------------------------
    random_rotation_angle: int | None = None
    """Maximum rotation angle (in degrees) for random rotation augmentation of input images."""

    color_jitter_params: dict[str, float] | None = None
    """
    Parameters for color jitter augmentation on images.

    Expected keys include:
      - "brightness": float
      - "contrast": float
      - "saturation": float
      - "hue": float
    Example: {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1}

    If None, applying the default color jitter augmentation from the pretrained model.
    """
    # ----------------------------------------------------------------------------------------------

    # Training Configuration -----------------------------------------------------------------------
    global_batch_size: int = 64
    """Total effective batch size across all GPUs and accumulation steps."""

    dataloader_num_workers: int = 2
    """Number of parallel worker processes used for data loading."""

    learning_rate: float = 1e-4
    """Initial learning rate for optimizer."""

    lr_scheduler_type: str = "cosine"
    """Learning rate scheduler type."""

    gradient_accumulation_steps: int = 1
    """Number of forward passes to accumulate before performing a backward/update step."""

    max_grad_norm: float = 1.0
    """Gradient clipping threshold (passed to ``TrainingArguments.max_grad_norm``).
    The value existed on ``TrainingConfig`` but was never exposed on the CLI
    until Issue #22 — same silent-no-op pattern as PR #15's RTC fields."""

    optim: str = "adamw_torch_fused"
    """Optimizer choice forwarded to ``TrainingArguments.optim``. Default is
    the fused AdamW kernel (matches ``TrainingConfig`` default). Assembly
    used to hard-overwrite this to ``adamw_torch``, discarding both the
    fused default and any CLI value; Issue #22 removed that hard-wire."""

    output_dir: str = "./outputs"
    """Directory where model checkpoints, logs, and outputs are saved."""

    save_steps: int = 1000
    """Frequency (in training steps) at which to save checkpoints."""

    save_total_limit: int = 5
    """Maximum number of checkpoints to keep before older ones are deleted."""

    num_gpus: int = 1
    """Number of GPUs available for distributed or single-node training."""

    use_wandb: bool = False
    """
    If True, log metrics and artifacts to Weights & Biases (wandb).
    You need to login to wandb to view the logs.
    """

    experiment_name: str = "debug"

    wandb_project: str = "RLDX-1"
    """Weights & Biases project name."""

    max_steps: int = 10000
    """Total number of training steps to run before stopping."""

    new_param_warmup_steps: int = 0
    """Number of initial training steps where only newly-added parameters (from motion module, memory,
    physics modules) are trained. Set to 0 to disable (all trainable params from step 0)."""

    weight_decay: float = 1e-5
    """Weight decay coefficient for optimizer (L2 regularization)."""

    warmup_ratio: float = 0.05
    """Proportion of total training steps used for learning rate warm-up."""

    dataset_mode: str = "sharded"
    """Dataset loading mode: 'sharded' (default) or 'standard'.
    'sharded': pre-shards episodes for background prefetch (episode_sampling_rate applies).
    'standard': map-style random-access dataset; all valid steps are included, no sharding."""

    shard_size: int = 2**10
    """Size of the shard to use for the dataset during preloading (sharded mode only)."""

    episode_sampling_rate: float = 0.1
    """Sampling rate for the episodes (sharded mode only)."""

    num_shards_per_epoch: int = int(1e5)
    """Number of shards to use for the dataset. reduce this number if vram is limited (sharded mode only)."""
    # ----------------------------------------------------------------------------------------------
