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
# This file has been modified from the original NVIDIA Isaac GR00T N1.7.
# Original source: https://github.com/NVIDIA/Isaac-GR00T

import copy
import importlib
import json
from pathlib import Path
import shutil
import sys

from termcolor import colored
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from rldx.utils.dist import rank_zero_print as _print


# NOTE(MK): Model architecture keys based on `RLDXConfig`, excluding:
#   - Training-only keys (e.g., `tune_llm`, `state_dropout_prob`)
#   - Processing/augmentation keys (e.g., `image_crop_size`, `color_jitter_params`)
#   - Runtime keys (e.g., `model_dtype`, `load_bf16`)
#   - Inference keys (e.g., `num_inference_timesteps`, `noise_beta_alpha/beta/s`)
MODEL_ARCHITECTURE_KEYS = [
    "model_type",
    "model_name",
    "backbone_model_type",
    "backbone_embedding_dim",
    "n_cog_tokens",
    "use_memory",
    "memory_length",
    "memory_stride",
    "memory_n_cog_tokens",
    "concat_memory",
    "use_video",
    "video_length",
    "video_stride",
    "use_motion",
    "physics_dropout_prob",
    "hidden_size",
    "input_embedding_dim",
    "max_state_dim",
    "max_action_dim",
    "action_horizon",
    "select_layer",
    "max_num_embodiments",
]


def snapshot_model_config(config_model) -> dict:
    """Take a deep-copy snapshot of architecture-relevant fields from a model config."""
    return {key: copy.deepcopy(getattr(config_model, key, None)) for key in MODEL_ARCHITECTURE_KEYS}


def compare_model_configs(ckpt_snapshot: dict, final_cfg) -> dict:
    """Compare checkpoint model config snapshot with final model config.

    Returns a dict of {key: (old_val, new_val)} for any fields that differ.
    Logs differences as warnings (CLI overrides always take effect).
    """
    diffs = {}
    for key in MODEL_ARCHITECTURE_KEYS:
        ckpt_val = ckpt_snapshot.get(key)
        final_val = getattr(final_cfg, key, None)
        if ckpt_val != final_val:
            diffs[key] = (ckpt_val, final_val)

    if not diffs:
        _print(colored("[i] No model architecture changes detected from checkpoint.", "green"))
        return diffs

    msg_lines = ["[w] Model config differences (checkpoint -> CLI override):"]
    for key, (old, new) in diffs.items():
        msg_lines.append(f"  {key}: {old} -> {new}")
    _print(colored("\n".join(msg_lines), "yellow"))
    return diffs


def resolve_backbone_path(cli_backbone_path, base_model_path, ckpt_model_snapshot):
    """Resolve backbone path from CLI, checkpoint snapshot, or HF ``config.json``.

    Resolution order:
      1. ``--backbone-path`` (explicit CLI override).
      2. ``experiment_cfg/config.yaml`` snapshot's ``model_name`` (yaml-shipped checkpoints).
      3. ``<base_model_path>/config.json``'s ``model_name`` (HF-style checkpoints that
         only ship a PretrainedConfig — e.g. ``RLWRLD/RLDX-1-PT``). Logs a warning
         so the fallback is never silent.
      4. ``Qwen/Qwen3-VL-8B-Instruct`` default — only when ``base_model_path is None``
         (i.e. genuine from-scratch). Resolving from a *checkpoint* whose backbone
         we cannot identify raises instead of silently re-initialising the input
         embedding / lm_head against vanilla Qwen.
    """
    if cli_backbone_path is not None:
        _print(f"Using backbone path from CLI: {cli_backbone_path}")
        return cli_backbone_path
    if ckpt_model_snapshot is not None:
        snap_name = ckpt_model_snapshot.get("model_name")
        if snap_name:
            _print(f"[i] Backbone path not specified via CLI. Using checkpoint value: {snap_name}")
            return snap_name
    if base_model_path is not None:
        cfg_json = Path(base_model_path) / "config.json"
        if cfg_json.exists():
            try:
                cfg_name = json.loads(cfg_json.read_text()).get("model_name")
            except json.JSONDecodeError:
                cfg_name = None
            if cfg_name:
                _print(
                    colored(
                        f"[i] experiment_cfg/config.yaml not present at {base_model_path}; "
                        f"using model_name from config.json: {cfg_name}",
                        "yellow",
                    )
                )
                return cfg_name
        raise RuntimeError(
            f"Cannot resolve backbone for checkpoint at {base_model_path}. "
            "Pass --backbone-path explicitly, or ensure the checkpoint ships "
            "either experiment_cfg/config.yaml or a config.json with model_name. "
            "Refusing to silently fall back to vanilla Qwen/Qwen3-VL-8B-Instruct, "
            "which would re-initialise the input embedding and lm_head against "
            "the wrong vocabulary."
        )
    path = "Qwen/Qwen3-VL-8B-Instruct"
    _print(f"\n[i] Using default backbone path: {path}")
    return path


def load_modality_config(modality_config_path: str):
    """Import a user-provided modality config Python file so its registrations take effect."""
    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        _print(f"\n[i] Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


# ---------------------------------------------------------------------------
# Trainer callbacks
# ---------------------------------------------------------------------------


class CheckpointFormatCallback(TrainerCallback):
    """This callback format checkpoint to make them standalone. For now, it copies all config
    files to /checkpoint-{step}/experiment_cfg/:
    - conf.yaml
    - initial_actions.npz
    - metadata.json
    """

    def __init__(
        self, run_name: str, exp_cfg_dir: Path | None = None, processor_dir: Path | None = None
    ):
        """
        Args:
            run_name: Name of the experiment run
            exp_cfg_dir: Path to the directory containing all experiment metadata
        """
        self.exp_cfg_dir = exp_cfg_dir
        self.processor_dir = processor_dir

    def on_save(self, args, state, control, **kwargs):
        """Called after the trainer saves a checkpoint."""
        if state.is_world_process_zero:
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"

            # Copy experiment config directory if provided
            if self.exp_cfg_dir is not None:
                exp_cfg_dst = checkpoint_dir / self.exp_cfg_dir.name
                if self.exp_cfg_dir.exists():
                    print(
                        f"Copying experiment config directory {self.exp_cfg_dir} to {exp_cfg_dst}"
                    )
                    shutil.copytree(self.exp_cfg_dir, exp_cfg_dst, dirs_exist_ok=True)

            # Copy processor directory if provided.
            # Destination is checkpoint_dir/"processor" (subdir) so the layout matches
            # what RLDXPolicy loads via AutoProcessor.from_pretrained(model_dir / "processor").
            if self.processor_dir is not None:
                if self.processor_dir.exists():
                    processor_dst = checkpoint_dir / "processor"
                    print(f"Copying processor directory {self.processor_dir} to {processor_dst}")
                    shutil.copytree(self.processor_dir, processor_dst, dirs_exist_ok=True)

            # Copy wandb_config.json if provided
            wandb_config_src = Path(args.output_dir) / "wandb_config.json"
            wandb_config_dst = checkpoint_dir / "wandb_config.json"
            if wandb_config_src.exists():
                print(f"Copying wandb_config.json from {wandb_config_src} to {wandb_config_dst}")
                shutil.copy2(wandb_config_src, wandb_config_dst)


class BestMetricCheckpointCallback(TrainerCallback):
    """This callback saves the best checkpoint based on the metric."""

    def __init__(
        self, metric_name: str, greater_is_better: bool = True, exp_cfg_dir: Path | None = None
    ):
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_metric = -float("inf") if greater_is_better else float("inf")
        self.exp_cfg_dir = exp_cfg_dir
        self._best_checkpoint_dir = None

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics,
        model,
        **kwargs,
    ):
        if state.is_world_process_zero and metrics is not None:
            current_metric = metrics.get(self.metric_name, None)
            if current_metric is not None:
                is_better = (
                    self.greater_is_better
                    if current_metric > self.best_metric
                    else not self.greater_is_better
                )
                if is_better:
                    self.best_metric = current_metric
                    best_checkpoint_dir = (
                        Path(args.output_dir)
                        / f"checkpoint-{state.global_step}-best-{self.metric_name}_{current_metric}"
                    )
                    best_checkpoint_dir.mkdir(exist_ok=True)
                    model.save_pretrained(best_checkpoint_dir)
                    # Copy experiment config directory if provided
                    if self.exp_cfg_dir is not None:
                        exp_cfg_dst = best_checkpoint_dir / self.exp_cfg_dir.name
                        if self.exp_cfg_dir.exists():
                            print(
                                f"Copying experiment config directory {self.exp_cfg_dir} to {exp_cfg_dst}"
                            )
                            shutil.copytree(self.exp_cfg_dir, exp_cfg_dst, dirs_exist_ok=True)

                    print(
                        f"Best checkpoint saved to {best_checkpoint_dir} with metric {self.metric_name} = {current_metric}"
                    )

                    if (
                        self._best_checkpoint_dir is not None
                        and Path(self._best_checkpoint_dir).exists()
                    ):
                        shutil.rmtree(self._best_checkpoint_dir)

                    self._best_checkpoint_dir = str(best_checkpoint_dir)


class NewParamWarmupCallback(TrainerCallback):
    """Warmup stage: train only newly-added parameters for the first N steps.

    During warmup, all originally-trainable parameters that are NOT in `new_param_names`
    are temporarily frozen. After `warmup_steps`, they are unfrozen and the optimizer +
    LR scheduler are recreated to include the newly-unfrozen parameters.
    """

    def __init__(self, warmup_steps: int, new_param_names: set[str], trainer):
        self.warmup_steps = warmup_steps
        self.new_param_names = new_param_names
        self.trainer = trainer
        self._original_requires_grad: dict[str, bool] = {}
        self._warmup_active = True

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.warmup_steps <= 0:
            return

        # Save original requires_grad state and freeze non-new params
        n_frozen = 0
        n_new_trainable = 0
        for name, param in model.named_parameters():
            self._original_requires_grad[name] = param.requires_grad
            if param.requires_grad and name not in self.new_param_names:
                param.requires_grad_(False)
                n_frozen += 1
            elif param.requires_grad and name in self.new_param_names:
                n_new_trainable += 1

        _print(
            colored(
                f"\n[NewParamWarmup] Warmup active for {self.warmup_steps} steps: "
                f"{n_new_trainable} new params trainable, {n_frozen} existing params frozen",
                "cyan",
            )
        )

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not self._warmup_active or state.global_step < self.warmup_steps:
            return

        # Warmup complete — restore original requires_grad
        n_unfrozen = 0
        for name, param in model.named_parameters():
            original = self._original_requires_grad.get(name, False)
            if original and not param.requires_grad:
                param.requires_grad_(True)
                n_unfrozen += 1

        remaining_steps = args.max_steps - state.global_step
        self.trainer.create_optimizer_and_scheduler(num_training_steps=remaining_steps)

        self._warmup_active = False
        _print(
            colored(
                f"\n[NewParamWarmup] Step {state.global_step}: warmup complete, "
                f"{n_unfrozen} params unfrozen, optimizer will be recreated",
                "cyan",
            )
        )


class MossGradientCheckCallback(TrainerCallback):
    """Log motion module parameter gradient norms at step 1, 5, and every 50 steps."""

    def __init__(self, log_steps=(1, 5)):
        self.log_steps = set(log_steps)
        self.log_interval = 50

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step not in self.log_steps and step % self.log_interval != 0:
            return

        moss_grads = {}
        moss_no_grad = []
        for name, param in model.named_parameters():
            if "motion" not in name.lower():
                continue
            if not param.requires_grad:
                continue
            if param.grad is not None:
                moss_grads[name] = param.grad.norm().item()
            else:
                moss_no_grad.append(name)

        if moss_grads or moss_no_grad:
            _print(colored(f"\n[motion module Gradient Check] Step {step}:", "yellow"))
            for name, norm in moss_grads.items():
                short = name.split(".")[-2] + "." + name.split(".")[-1]
                _print(f"  {short}: grad_norm={norm:.6f}")
            if moss_no_grad:
                _print(
                    colored(
                        f"  WARNING: {len(moss_no_grad)} motion module params have grad=None", "red"
                    )
                )
                for name in moss_no_grad[:5]:
                    _print(f"    - {name}")
