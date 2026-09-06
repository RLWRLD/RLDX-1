# SPDX-License-Identifier: Apache-2.0
"""
PolicyLoader — RLDX checkpoint + processor + config loading, extracted from
RLDXPolicy.__init__.

Before this extraction, ~240 LOC of loading/config logic lived inline in
``RLDXPolicy.__init__`` — mixed with session init, validator construction,
and general setup. Problems:
  - Hard to test loading in isolation (can't bypass __init__ without
    stubbing dozens of fields).
  - Fallback path (primary from_pretrained fails → rebuild from config +
    safetensors) was silently buried in a try/except.
  - Post-load mutations on model/processor (RTC override, physics config
    injection, video delta adjustment) were scattered and hard to trace.

PolicyLoader owns these concerns:
  1. Resolve model_dir (local or HF snapshot download)
  2. Load model (primary from_pretrained + fallback safetensors + key remap)
  3. Apply model-level tweaks (sample_timestep_from_beta_dist, denoising)
  4. Load processor (subdir-aware) + inject physics / conversation_image_first
  5. Resolve embodiment-specific modality_configs + filter physics
     delta_indices for inference
  6. Validate language config (single key, single delta)
  7. Apply RTC overrides onto model.config + sync action_model._rtc
  8. Compute use_memory flag + adjust video delta_indices for memory path

Returns a LoaderResult dataclass that RLDXPolicy.__init__ unpacks into
self.* fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor

from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.interfaces import BaseProcessor


# ----------------------------------------------------------------------------
# Inputs / outputs
# ----------------------------------------------------------------------------


@dataclass
class RTCOverrides:
    """Optional CLI/kwarg RTC overrides applied post-load onto model.config."""

    mode: str | None = None
    delay: int | None = None
    exec_horizon: int | None = None
    jacobian_beta: float | None = None
    jacobian_steps_only: int | None = None


@dataclass
class LoaderResult:
    """Everything RLDXPolicy needs from a successful load.

    Field groups:
      - model / processor: runtime objects (stateful, mutated by Loader)
      - modality_configs / embodiment_tag / collate_fn / language_key:
        resolved observation / action schema
      - physics_keys: model-configured physics streams (for Validator)
      - rtc_inference_mode / rtc_inference_delay / rtc_exec_horizon /
        rtc_enabled: resolved RTC runtime knobs (CLI overrides already applied)
      - use_memory: resolved memory flag (reflects model.config after
        deactivate_memory override)
    """

    model: Any
    processor: BaseProcessor
    embodiment_tag: EmbodimentTag
    modality_configs: dict
    collate_fn: Any
    language_key: str
    physics_keys: list[str]
    rtc_inference_mode: str
    rtc_inference_delay: int
    rtc_exec_horizon: int
    rtc_enabled: bool
    use_memory: bool


# ----------------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------------


class PolicyLoader:
    """Stateless loader — all inputs via ``load()`` kwargs, all outputs via
    LoaderResult. No instance state."""

    @staticmethod
    def load(
        *,
        embodiment_tag: EmbodimentTag,
        model_path: str,
        device: int | str,
        deactivate_memory: bool = False,
        sample_timestep_from_beta_dist: bool = False,
        denoising_timesteps: list[float] | None = None,
        rtc_overrides: RTCOverrides | None = None,
    ) -> LoaderResult:
        """Full load pipeline. See module docstring for phase summary."""
        model_dir = _resolve_model_dir(model_path)
        model = _load_model(model_dir, device, deactivate_memory)
        _apply_model_tweaks(model, sample_timestep_from_beta_dist, denoising_timesteps)

        processor = _load_processor(model_dir, model)

        modality_configs = processor.get_modality_configs()[embodiment_tag.value]
        collate_fn = processor.collator

        physics_keys = list(getattr(model.config, "physics_keys", []) or [])
        _filter_physics_delta_indices(modality_configs, model, physics_keys)

        language_key = _resolve_language_key(modality_configs)

        rtc_inference_mode, rtc_inference_delay, rtc_exec_horizon, rtc_enabled = (
            _apply_rtc_overrides(model, rtc_overrides or RTCOverrides())
        )

        use_memory = _apply_memory_config(model, modality_configs)

        return LoaderResult(
            model=model,
            processor=processor,
            embodiment_tag=embodiment_tag,
            modality_configs=modality_configs,
            collate_fn=collate_fn,
            language_key=language_key,
            physics_keys=physics_keys,
            rtc_inference_mode=rtc_inference_mode,
            rtc_inference_delay=rtc_inference_delay,
            rtc_exec_horizon=rtc_exec_horizon,
            rtc_enabled=rtc_enabled,
            use_memory=use_memory,
        )


# ----------------------------------------------------------------------------
# Phase helpers
# ----------------------------------------------------------------------------


def _resolve_model_dir(model_path: str) -> Path:
    model_dir = Path(model_path)
    if not model_dir.exists():
        model_dir = Path(snapshot_download(model_path))
    return model_dir


def _load_model(
    model_dir: Path,
    device: int | str,
    deactivate_memory: bool,
) -> Any:
    """Load model + optionally disable memory inference; return in eval mode."""
    import rldx.model  # noqa: F401

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

    if getattr(config, "use_memory", False) and deactivate_memory:
        print("[RLDXPolicy] deactivate_memory=True: disabling memory for inference...")
        config.use_memory = False
        config.concat_memory = False

    model = AutoModel.from_pretrained(model_dir, device_map=device, torch_dtype=torch.bfloat16)
    model.eval()
    return model


def _apply_model_tweaks(
    model: Any,
    sample_timestep_from_beta_dist: bool,
    denoising_timesteps: list[float] | None,
) -> None:
    """Post-load mutations for inference-specific sampling knobs."""
    if sample_timestep_from_beta_dist and hasattr(model, "action_model"):
        model.action_model.sample_timestep_from_beta_dist = True
        print("[Policy] Set inference timestep sampling to Beta distribution")
    if denoising_timesteps is not None and hasattr(model, "action_model"):
        model.action_model.denoising_timesteps = denoising_timesteps
        print(f"[Policy] Set fixed denoising timesteps: {denoising_timesteps}")


def _load_processor(model_dir: Path, model: Any) -> BaseProcessor:
    """Load processor (subdir-aware) + inject physics config + image-first flag."""
    import rldx.model  # noqa: F401

    _processor_subdir = model_dir / "processor"
    _processor_path = _processor_subdir if _processor_subdir.exists() else model_dir
    processor: BaseProcessor = AutoProcessor.from_pretrained(_processor_path)

    # Fallback: inject physics config from model config if processor lacks it
    if not processor.physics_keys and getattr(model.config, "physics_keys", None):
        processor.physics_keys = model.config.physics_keys
        processor.physics_dims = getattr(model.config, "physics_dims", [])
        processor.allow_missing_physics = getattr(model.config, "allow_missing_physics", False)
        print(
            f"[RLDXPolicy] Injected physics config from model: "
            f"keys={processor.physics_keys}, dims={processor.physics_dims}"
        )

    if getattr(model.config, "conversation_image_first", False):
        processor.conversation_image_first = True
        print("[RLDXPolicy] Set conversation_image_first to True")

    processor.eval()
    return processor


def _filter_physics_delta_indices(
    modality_configs: dict,
    model: Any,
    physics_keys: list[str],
) -> None:
    """Filter physics delta_indices to inference-appropriate subset.

    Conditioning frames (d<=0) are kept; future frames are dropped (model
    generates those from noise). If no conditioning frames remain, drop the
    physics key from modality_configs entirely (fut-only mode).
    """
    physics_hist_len = getattr(model.config, "physics_hist_len", None)
    for pk in physics_keys:
        if pk not in modality_configs:
            continue
        full_delta = modality_configs[pk].delta_indices
        if physics_hist_len is None:
            cond_delta = [d for d in full_delta if d <= 0]
        else:
            cond_delta = full_delta[:physics_hist_len]

        if len(cond_delta) > 0:
            modality_configs[pk].delta_indices = cond_delta
            print(
                f"[RLDXPolicy] Physics '{pk}' inference delta_indices: "
                f"{cond_delta} ({len(cond_delta)} frames, hist only)"
            )
        else:
            del modality_configs[pk]
            print(
                f"[RLDXPolicy] Physics '{pk}' fut-only mode: no client input "
                f"needed (model generates future from noise)"
            )


def _resolve_language_key(modality_configs: dict) -> str:
    """Validate language modality assumptions + return the single language key."""
    language_keys = modality_configs["language"].modality_keys
    language_delta_indices = modality_configs["language"].delta_indices
    assert len(language_delta_indices) == 1, "Only one language delta index is supported"
    assert len(language_keys) == 1, "Only one language key is supported"
    return language_keys[0]


def _apply_rtc_overrides(
    model: Any,
    overrides: RTCOverrides,
) -> tuple[str, int, int, bool]:
    """Write overrides onto model.config + sync action_model._rtc.

    Returns (mode, delay, exec_horizon, enabled) resolved after overrides.
    """
    if overrides.mode is not None:
        model.config.rtc_inference_mode = str(overrides.mode)
    if overrides.delay is not None:
        model.config.rtc_inference_delay = int(overrides.delay)
    if overrides.exec_horizon is not None:
        model.config.rtc_inference_exec_horizon = int(overrides.exec_horizon)
    if overrides.jacobian_beta is not None:
        model.config.rtc_jacobian_beta = float(overrides.jacobian_beta)
    if overrides.jacobian_steps_only is not None:
        model.config.rtc_jacobian_steps_only = int(overrides.jacobian_steps_only)

    # Sync action_model's cached RTCConfig
    if hasattr(model, "action_model") and hasattr(model.action_model, "_rtc"):
        from rldx.model.modules.action_model.rtc import rtc_config_from_rldx

        model.action_model._rtc = rtc_config_from_rldx(model.config)
        model.action_model._rtc.validate(model.config.action_horizon)

    mode = str(getattr(model.config, "rtc_inference_mode", "none"))
    delay = int(getattr(model.config, "rtc_inference_delay", 0) or 0)
    _action_horizon = int(getattr(model.config, "action_horizon", 0))
    _exec = int(getattr(model.config, "rtc_inference_exec_horizon", 0) or 0)
    exec_horizon = _exec if _exec > 0 else max(_action_horizon - delay, 0)
    enabled = mode != "none" and delay > 0

    if enabled:
        print(f"[RLDXPolicy] RTC enabled: mode={mode} delay={delay} exec_horizon={exec_horizon}")

    return mode, delay, exec_horizon, enabled


def _compute_inference_video_delta_indices(video_length: int, video_stride: int) -> list[int]:
    """Frame-delta indices for recurrent video inference.

    Mirrors the training-time stride layout in
    ``rldx/experiment/features/video.py::VideoFeature.apply``. Anchors the
    most-recent frame at delta=0 and walks backwards by ``video_stride``
    action-steps. Returning a list (rather than the training-side set)
    preserves order so ``modality_configs["video"].delta_indices`` stays
    chronological.

    Raises ``ValueError`` for ``video_length < 1`` (degenerate empty
    window) or ``video_stride < 1`` (would yield duplicate frame indices
    and silently feed the model the same frame repeatedly).
    """
    vl = int(video_length)
    vs = int(video_stride)
    if vl < 1:
        raise ValueError(f"video_length must be >= 1, got {vl}")
    if vs < 1:
        raise ValueError(
            f"video_stride must be >= 1 (got {vs}); stride 0 would produce "
            f"duplicate frame indices and silently sample the same frame."
        )
    return [(i - (vl - 1)) * vs for i in range(vl)]


def _apply_memory_config(model: Any, modality_configs: dict) -> bool:
    """Resolve use_memory + adjust video delta_indices for memory inference path.

    Memory-enabled models require recurrent single-frame inference, so video
    delta_indices collapse to [0] (or strided look-back window when
    ``use_video`` is on). Reads ``video_stride`` from ``model.config`` so
    inference frame deltas align with the training side
    (``features/video.py``); falls back to 2 for older checkpoints
    that lack the field.
    """
    use_memory = getattr(model.config, "use_memory", False)
    if not use_memory:
        return False

    memory_length = getattr(model.config, "memory_length", 4)
    memory_n_cog_tokens = getattr(model.config, "memory_n_cog_tokens", None)
    print(
        f"[RLDXPolicy] Memory enabled with memory_length={memory_length}, "
        f"memory_n_cog_tokens={memory_n_cog_tokens}"
    )

    use_video = getattr(model.config, "use_video", False)
    if use_video:
        video_length = int(getattr(model.config, "video_length", 4))
        video_stride = int(getattr(model.config, "video_stride", 2))
        video_delta_indices = _compute_inference_video_delta_indices(video_length, video_stride)
        print(
            f"[RLDXPolicy] Memory & Video enabled. Automatically setting video "
            f"delta indices to {video_delta_indices} "
            f"(video_length={video_length}, video_stride={video_stride}) "
            f"for recurrent inference."
        )
        modality_configs["video"].delta_indices = video_delta_indices
    else:
        print(
            "[RLDXPolicy] Automatically setting video delta indices to [0] for recurrent inference."
        )
        modality_configs["video"].delta_indices = [0]

    return True
