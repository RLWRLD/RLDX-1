"""Pure-function assembly of CLI + environment inputs into a RunConfig.

I/O (HF download, YAML reading, file existence) and the trainer kickoff stay
in the launcher; this module only mutates config dataclasses.
"""

import copy
from dataclasses import dataclass
import os
from typing import Optional

from rldx.configs.base_config import Config, get_config
from rldx.configs.data.dataset_mix import dataset_mix
from rldx.configs.train_config import TrainConfig
from rldx.experiment.features import FEATURES, AssemblyContext, _check_dependencies
from rldx.experiment.utils import compare_model_configs, resolve_backbone_path
from rldx.utils.dist import rank_zero_print as _print


# ============================================================================
# Pure helpers (moved verbatim from launch_train.py)
# ============================================================================


def build_dataset_specs(config: TrainConfig, embodiment_tag: str) -> list[dict]:
    """Build dataset specs for finetune mode (config.data.datasets).

    Accepted forms:
      - dataset_path -> single dataset
      - dataset_paths (+ optional dataset_mix_ratios) -> multi-dataset
    """
    if config.dataset_paths is not None and len(config.dataset_paths) > 0:
        dataset_paths = config.dataset_paths
        if config.dataset_mix_ratios is None:
            mix_ratios = [1.0] * len(dataset_paths)
        else:
            mix_ratios = config.dataset_mix_ratios
            if len(mix_ratios) != len(dataset_paths):
                raise ValueError(
                    f"dataset_mix_ratios length ({len(mix_ratios)}) must match "
                    f"dataset_paths length ({len(dataset_paths)})"
                )
        if any(r <= 0 for r in mix_ratios):
            raise ValueError("All dataset_mix_ratios must be > 0")
        return [
            {
                "dataset_paths": [path],
                "mix_ratio": ratio,
                "embodiment_tag": embodiment_tag,
            }
            for path, ratio in zip(dataset_paths, mix_ratios)
        ]

    if config.dataset_path is None:
        raise ValueError("Either dataset_path or dataset_paths must be provided for finetune mode")
    return [
        {
            "dataset_paths": [config.dataset_path],
            "mix_ratio": 1.0,
            "embodiment_tag": embodiment_tag,
        }
    ]


def build_pt_dataset_specs(config: TrainConfig) -> list[dict]:
    """Build dataset specs for pretrain mode from pt_dataset_root and pt_dataset_mix."""
    pt_dataset_mix_config = dataset_mix[config.pt_dataset_mix]
    return [
        {
            "dataset_paths": [os.path.join(config.pt_dataset_root, d["dataset_name"])],
            "mix_ratio": d["mix_ratio"],
            "embodiment_tag": d["embodiment_tag"].value,
        }
        for d in pt_dataset_mix_config
    ]


def _assert_dataset_config_mutually_exclusive(config: TrainConfig) -> None:
    """Assert that finetune and pretrain dataset configs are not used simultaneously."""
    use_ft = (config.dataset_path is not None) or (
        config.dataset_paths is not None and len(config.dataset_paths) > 0
    )
    use_pt = (config.pt_dataset_root is not None) and (config.pt_dataset_mix is not None)
    if use_ft and use_pt:
        raise ValueError(
            "Finetune and pretrain dataset configs cannot be used simultaneously. "
            "Use either (dataset_path or dataset_paths) for finetune, "
            "or (pt_dataset_root + pt_dataset_mix) for pretrain."
        )
    if not use_ft and not use_pt:
        raise ValueError(
            "Must specify dataset config for either finetune or pretrain mode. "
            "Finetune: dataset_path or dataset_paths. "
            "Pretrain: pt_dataset_root and pt_dataset_mix."
        )


# ============================================================================
# Assembly inputs — explicit contract
# ============================================================================


@dataclass(frozen=True)
class AssemblyInputs:
    """All data `assemble_run_config` needs. Caller owns I/O (HF download, YAML read)."""

    cli: TrainConfig
    datasets: list[dict]
    base_model_path: Optional[str] = None
    """Resolved local path to the base checkpoint (``None`` for from-scratch).
    Used by `_apply_training_overrides` to set
    ``run_config.training.start_from_checkpoint``."""
    loaded_yaml_config: Optional[Config] = None
    """Populated when base_model_path points to a checkpoint shipping
    experiment_cfg/config.yaml. None = from-scratch."""
    loaded_ckpt_model_snapshot: Optional[dict] = None
    """Snapshot of loaded_yaml_config.model before CLI overrides.
    Used only for compare_model_configs warning. None = skip the comparison."""


# ============================================================================
# Per-block appliers (each mutates run_config in place; returns None)
# ============================================================================


def _apply_cli_model_overrides(
    cli: TrainConfig,
    run_config: Config,
    base_model_path: Optional[str],
    loaded_ckpt_model_snapshot: Optional[dict],
) -> None:
    """Replicate launch_train.py L177-204."""
    run_config.model.model_type = "RLDX-1"
    run_config.model.tune_llm = cli.tune_llm
    run_config.model.tune_visual = cli.tune_visual
    run_config.model.tune_projector = cli.tune_projector
    run_config.model.tune_diffusion_model = cli.tune_diffusion_model
    # Action model LoRA. Mirror every CLI knob onto the model config so
    # the saved config.json records the values the user actually trained
    # with — skipping any of these would turn the CLI flag into a silent
    # no-op.
    run_config.model.action_model_use_lora = cli.action_model_use_lora
    run_config.model.action_model_lora_rank = cli.action_model_lora_rank
    run_config.model.action_model_lora_alpha = cli.action_model_lora_alpha
    run_config.model.action_model_lora_dropout = cli.action_model_lora_dropout
    if cli.action_model_use_lora and cli.tune_diffusion_model:
        _print(
            "[i] action_model_use_lora=True: overriding tune_diffusion_model "
            "(LoRA adapters control the MSAT trainable state)."
        )
    # Backbone (Qwen3 LLM) LoRA. Copy every CLI field onto run_config.model
    # so the saved config.json records the values the user actually trained
    # with, and warn when the user combines ``--backbone-use-lora`` with a
    # nonzero ``--tune-top-llm-layers`` (LoRA owns the LLM trainable surface;
    # tune_top_llm_layers is silently ignored under LoRA).
    run_config.model.backbone_use_lora = cli.backbone_use_lora
    run_config.model.backbone_lora_rank = cli.backbone_lora_rank
    run_config.model.backbone_lora_alpha = cli.backbone_lora_alpha
    run_config.model.backbone_lora_dropout = cli.backbone_lora_dropout
    run_config.model.backbone_lora_num_layers = cli.backbone_lora_num_layers
    if cli.backbone_use_lora and cli.tune_top_llm_layers > 0:
        _print(
            "[i] backbone_use_lora=True: ignoring tune_top_llm_layers "
            "(LoRA adapters control which LLM layers are adapted)."
        )
    run_config.model.state_dropout_prob = cli.state_dropout_prob
    run_config.model.image_max_area = cli.image_max_area
    run_config.model.image_resize_m = cli.image_resize_m
    run_config.model.random_crop_fraction = cli.random_crop_fraction
    run_config.model.random_rotation_angle = cli.random_rotation_angle
    run_config.model.color_jitter_params = cli.color_jitter_params
    # When backbone_use_lora is on, the LoRA adapters own the LLM trainable
    # surface; copy 0 onto the model config so the warning above matches
    # behaviour. Otherwise BackboneAdapter would mark the top-N base
    # weights ``requires_grad=True`` and adapter.py's bf16→fp32 pre-cast
    # would run, only for ``_apply_backbone_lora`` to immediately re-freeze
    # them — wasted work, a misleading saved config.json, and a no-op fp32
    # cast in ``_apply_backbone_lora`` since PEFT inherits fp32 from the
    # (already-promoted) base layer.
    run_config.model.tune_top_llm_layers = 0 if cli.backbone_use_lora else cli.tune_top_llm_layers
    run_config.model.freeze_cog_tokens = cli.freeze_cog_tokens
    run_config.model.general_embodiment_train_ratio = cli.general_embodiment_train_ratio
    run_config.model.conversation_image_first = cli.conversation_image_first

    run_config.model.model_name = resolve_backbone_path(
        cli.backbone_path, base_model_path, loaded_ckpt_model_snapshot
    )
    # backbone_model_type is fixed to "vtc_qwen3_vl" at RLDXConfig default.
    # The CLI does not expose --backbone-model-type and checkpoint values
    # are ignored here.
    run_config.model.n_cog_tokens = cli.n_cog_tokens
    _print(f"[i] n_cog_tokens: {cli.n_cog_tokens}")
    run_config.model.action_horizon = cli.action_horizon
    _print(f"[i] action_horizon: {cli.action_horizon}")

    # Plumb CLI fields onto run_config.model.
    #   * backbone_select_layer: opt-in override (None keeps the ckpt-loaded
    #     / RLDXConfig default of 18); unconditional copy would overwrite a
    #     valid ckpt value with None.
    #   * model_revision: unconditional copy since None means "use HEAD",
    #     which is the same semantic as the RLDXConfig default.
    if cli.backbone_select_layer is not None:
        run_config.model.select_layer = cli.backbone_select_layer
    run_config.model.model_revision = cli.model_revision

    # Propagate offloadable feature flags so the model is built with exactly
    # the CLI's choice, even when the loaded checkpoint's yaml has them True.
    # FeatureModule.apply() also sets these to True when their CLI flag is on,
    # but it never sets them to False — so we overwrite here from the CLI
    # side of the ground truth. Video is NOT in this list: ckpt-True + CLI-False
    # is already rejected by `_validate_features_match_ckpt`.
    run_config.model.use_memory = cli.use_memory
    run_config.model.use_motion = cli.use_motion
    run_config.model.use_physics = cli.use_physics

    # RTC: copy every CLI knob so the saved checkpoint config records the
    # values the user actually trained with. Missing this copy turns
    # `--rtc-training-max-delay 4` into a silent no-op (model.config keeps the
    # RLDXConfig default of 0).
    run_config.model.rtc_training_max_delay = cli.rtc_training_max_delay
    run_config.model.rtc_inference_mode = cli.rtc_inference_mode
    run_config.model.rtc_inference_delay = cli.rtc_inference_delay
    run_config.model.rtc_inference_exec_horizon = cli.rtc_inference_exec_horizon
    # rtc_jacobian_beta is intentionally not copied: it's an inference-time
    # guidance knob, not a training-time setting. See RLDXConfig comment.
    run_config.model.rtc_jacobian_steps_only = cli.rtc_jacobian_steps_only


# Features that can be offloaded from a pretrained checkpoint (model arch
# stays intact apart from the dropped stream, and strict-false state_dict load
# ignores the unused weight prefix). ``use_video`` is intentionally NOT here:
# the release codebase fixes video on (vanilla Qwen3 dropped, ``--use-video``
# CLI knob removed), so video is now an architectural invariant — any
# checkpoint trained with ``use_video=False`` is unsupported and rejected
# separately at the bottom of this function.
_OFFLOADABLE_FEATURES: tuple[tuple[str, str], ...] = (
    ("use_memory", "--use-memory"),
    ("use_motion", "--use-motion"),
    ("use_physics", "--use-physics"),
)


def _validate_features_match_ckpt(cli, loaded_yaml_config) -> None:
    """Reconcile checkpoint feature flags with CLI flags.

    Video is an architectural invariant in the release codebase — a
    pre-release ``use_video=False`` checkpoint is rejected with a clear
    error, since loading non-VTC weights into the VTC backbone would
    silently mismatch shapes deeper in the load. Memory / motion module /
    Physics are soft-offloadable: when the CLI omits the flag the stream is
    dropped from the build and its weights are discarded at load. Going the
    other way (ckpt off, CLI on) is always allowed.
    """
    if loaded_yaml_config is None:
        return  # from-scratch, no checkpoint to compare against.

    ckpt_model = loaded_yaml_config.model

    # Fatal: a checkpoint without VTC weights cannot be loaded into the
    # release-codebase VTC backbone. Default ``use_video=True`` here
    # matches the ``RLDXConfig`` default, so older yaml dumps that lack
    # the field are treated as VTC-shaped rather than rejected.
    if not getattr(ckpt_model, "use_video", True):
        raise ValueError(
            "Checkpoint was trained with use_video=False (vanilla Qwen3 "
            "backbone). The release codebase only supports the VTC backbone "
            "(vtc_qwen3_vl) — there is no CLI knob to drop video tokens. "
            "Use a base checkpoint trained with VTC (every released "
            "RLWRLD/RLDX-1-* checkpoint qualifies)."
        )

    # Soft: memory / motion / physics off → CLI wins. Log so the user isn't
    # surprised when the pretrained weights for those streams get dropped.
    offloaded = [
        (attr, flag)
        for attr, flag in _OFFLOADABLE_FEATURES
        if getattr(ckpt_model, attr, False) and not getattr(cli, attr, False)
    ]
    if offloaded:
        lines = [
            "[feature offload] Checkpoint has features enabled that the CLI disables. "
            "These streams will be dropped for this run (ckpt weights ignored):"
        ]
        for attr, flag in offloaded:
            lines.append(f"  - ckpt {attr}=True, CLI {flag} not set → disabled")
        _print("\n".join(lines))


def _validate_action_horizon_matches_modality(
    action_horizon: int,
    modality_configs: dict,
    used_embodiment_tags: set[str] | None = None,
) -> None:
    """Fail fast if a used embodiment's action.delta_indices length disagrees
    with action_horizon. Embodiments not in `used_embodiment_tags` are skipped;
    pass None to check every entry.
    """
    for emb_key, mods in modality_configs.items():
        if used_embodiment_tags is not None and emb_key not in used_embodiment_tags:
            continue
        if "action" not in mods:
            continue
        actual = len(mods["action"].delta_indices)
        if actual != action_horizon:
            raise ValueError(
                f"action_horizon={action_horizon} disagrees with modality_config "
                f"for embodiment '{emb_key}': action.delta_indices has length {actual}. "
                f"Fix either --action-horizon or the modality config so they match."
            )


# ============================================================================
# Orchestrator
# ============================================================================


def _load_base_config(inputs: AssemblyInputs) -> Config:
    """Build the base run_config prior to any CLI overrides.

    - loaded_yaml_config is not None: start from the deepcopied YAML'd Config
      (checkpoint shipped experiment_cfg/config.yaml).
    - loaded_yaml_config is None: start from a fresh `get_config("RLDX-1")`.

    In both cases we call load_dict({"data": ...}) to inject the CLI-built
    dataset specs and reset data.download_cache. Note that DataConfig's
    default_factory for modality_configs returns the shared module-level
    MODALITY_CONFIGS dict (data_config.py L50-52); since the feature-block
    appliers mutate delta_indices in place, we deepcopy modality_configs here
    to prevent state leaking across successive assemble_run_config calls.
    """
    if inputs.loaded_yaml_config is not None:
        run_config = copy.deepcopy(inputs.loaded_yaml_config)
    else:
        run_config = get_config("RLDX-1")

    run_config.load_dict({"data": {"download_cache": False, "datasets": inputs.datasets}})
    run_config.load_config_path = None
    run_config.data.modality_configs = copy.deepcopy(run_config.data.modality_configs)
    return run_config


def _apply_training_overrides(
    cli: TrainConfig,
    run_config: Config,
    base_model_path: Optional[str],
) -> None:
    """Apply CLI → run_config.training and run_config.data (non-modality) fields.

    Extracted from launch_train.py's inline post-assembly block. Separate from
    ``_apply_cli_model_overrides`` because these fields configure the Trainer
    and the dataloader, not the model architecture.
    """
    _print("\n2. Overwriting with CLI 'training' configs...")
    run_config.training.start_from_checkpoint = base_model_path if cli.base_model_path else None
    run_config.training.optim = cli.optim
    run_config.training.max_grad_norm = cli.max_grad_norm
    run_config.training.global_batch_size = cli.global_batch_size
    run_config.training.dataloader_num_workers = cli.dataloader_num_workers
    run_config.training.learning_rate = cli.learning_rate
    run_config.training.lr_scheduler_type = cli.lr_scheduler_type
    run_config.training.gradient_accumulation_steps = cli.gradient_accumulation_steps
    run_config.training.output_dir = cli.output_dir
    run_config.training.save_steps = cli.save_steps
    run_config.training.save_total_limit = cli.save_total_limit
    run_config.training.num_gpus = cli.num_gpus
    run_config.training.use_wandb = cli.use_wandb
    run_config.training.max_steps = cli.max_steps
    run_config.training.weight_decay = cli.weight_decay
    run_config.training.warmup_ratio = cli.warmup_ratio
    run_config.training.wandb_project = cli.wandb_project
    run_config.training.experiment_name = cli.experiment_name
    run_config.training.new_param_warmup_steps = cli.new_param_warmup_steps
    run_config.data.dataset_mode = cli.dataset_mode
    run_config.data.shard_size = cli.shard_size
    run_config.data.episode_sampling_rate = cli.episode_sampling_rate
    run_config.data.num_shards_per_epoch = cli.num_shards_per_epoch


def assemble_run_config(inputs: AssemblyInputs) -> Config:
    """Compose the run_config from CLI + (optional) loaded YAML/snapshot.

    Pure: no file I/O, no HF download, no Trainer start. Callers must have
    already resolved base_model_path, downloaded the model, and loaded any
    experiment_cfg/config.yaml.

    Flow:
      1. Load base Config (from YAML or defaults)
      2. Apply non-feature CLI model overrides (always)
      3. Check FeatureDependency across active features (T-E)
      4. Apply each active feature's mutation via the Registry
      5. Finalize per-modality delta_indices as anchor×stride cross-product (T-A/T-B)
      6. Compare vs checkpoint snapshot if provided
    """
    _print("\n1. Overwriting with CLI 'model' configs...")
    run_config = _load_base_config(inputs)
    _validate_features_match_ckpt(inputs.cli, inputs.loaded_yaml_config)
    _apply_cli_model_overrides(
        inputs.cli,
        run_config,
        inputs.base_model_path,
        inputs.loaded_ckpt_model_snapshot,
    )

    ctx = AssemblyContext(cli=inputs.cli, run_config=run_config)
    active = tuple(F for F in FEATURES if F.is_active(inputs.cli))
    _check_dependencies(active)
    for F in active:
        F.apply(ctx)
    ctx.finalize()

    _validate_action_horizon_matches_modality(
        inputs.cli.action_horizon,
        run_config.data.modality_configs,
        used_embodiment_tags={ds["embodiment_tag"] for ds in inputs.datasets},
    )

    if inputs.loaded_ckpt_model_snapshot is not None:
        compare_model_configs(inputs.loaded_ckpt_model_snapshot, run_config.model)

    _apply_training_overrides(inputs.cli, run_config, inputs.base_model_path)

    return run_config
