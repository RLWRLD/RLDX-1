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

import json
import logging
import os
from pathlib import Path
import warnings

from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from transformers import TrainingArguments, set_seed
import wandb

from rldx.configs.base_config import Config
from rldx.experiment.trainer import ProfCallback, RLDXTrainer
from rldx.experiment.utils import (
    BestMetricCheckpointCallback,
    CheckpointFormatCallback,
    NewParamWarmupCallback,
)
from rldx.model import MODEL_REGISTRY
from rldx.utils.dist import rank_zero_print as _print
from rldx.utils.initial_actions import INITIAL_ACTIONS_FILENAME, save_initial_actions


def setup_logging(debug: bool = False):
    """Configure logging."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    # Reduce verbosity of some libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)


def warn_configs(config: Config):
    # updates to batch size
    assert config.training.global_batch_size % config.training.num_gpus == 0, (
        "global_batch_size must be divisible by num_gpus"
    )

    if config.data.video_backend != "torchcodec":
        warnings.warn(
            "video_backend is not torchcodec. Only torchcodec will be supported in the future."
        )

    if config.training.batch_size is not None:
        warnings.warn(
            "batch_size will be deprecated in the future, please use global_batch_size instead. For now, this will override global_batch_size."
        )

    if config.training.warmup_steps > 0:
        warnings.warn(
            "warmup_steps will be deprecated in the future, please use warmup_ratio instead. For now, this will override warmup_ratio."
        )

    if (
        hasattr(config.model, "backbone_trainable_params_fp32")
        and not config.model.backbone_trainable_params_fp32
    ):
        warnings.warn(
            "backbone_trainable_params_fp32 is not True. This will be deprecated in the future."
        )

    if (
        getattr(config.model, "random_crop_fraction", None) is not None
        and not 0.0 < config.model.random_crop_fraction <= 1.0
    ):
        raise ValueError(
            f"random_crop_fraction must be in (0.0, 1.0], got {config.model.random_crop_fraction!r}"
        )


def run(config: Config):
    warn_configs(config)

    """Main training function."""
    # If using distributed training, initialize the process group
    if dist.is_initialized():
        global_rank = dist.get_rank()
    elif "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        import datetime

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=72000))
        # only meaningful for torchrun, for ray it is always 0
        global_rank = dist.get_rank()
    else:
        local_rank = 0
        global_rank = 0

    # Setup
    setup_logging()
    set_seed(config.data.seed)

    # Validate config
    config.validate()

    # Create output directory
    if config.training.experiment_name is None:
        output_dir = Path(config.training.output_dir)
        experiment_name = output_dir.name
    else:
        output_dir = Path(config.training.output_dir) / config.training.experiment_name
        experiment_name = config.training.experiment_name

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    save_cfg_dir = output_dir / "experiment_cfg"
    processor_dir = output_dir / "processor"
    config.save(save_cfg_dir / "config.yaml")  # For logging

    if hasattr(config.model, "_fill_missing_defaults"):
        config.model._fill_missing_defaults()

    omegaconf_config = OmegaConf.create(config.__dict__)
    omegaconf_config["max_steps"] = config.training.max_steps
    omegaconf_config["save_steps"] = config.training.save_steps
    OmegaConf.save(omegaconf_config, save_cfg_dir / "conf.yaml", resolve=True)
    wandb_config_file = output_dir / "wandb_config.json"
    with open(wandb_config_file, "w") as f:
        json.dump(
            {
                "project": config.training.wandb_project,
                "run_id": experiment_name,
            },
            f,
        )

    _print(f"\n[i] Saved config to {save_cfg_dir}")

    # Initialize wandb if configured, but only on the main process
    if config.training.use_wandb and global_rank == 0:
        # Add git commit hash and version info to config
        config_dict = {
            **config.__dict__,
            "git_commit_hash": os.environ.get("GROOT_COMMIT_HASH", "unknown"),
        }

        wandb.init(
            project=config.training.wandb_project,
            name=experiment_name,
            config=config_dict,
            tags=[config.data.mode],
        )

    # Setup model training pipeline.
    pipeline = MODEL_REGISTRY.get(type(config.model))(config, save_cfg_dir)  # RLDXPipeline
    pipeline.setup()  # create model, dataset, and data collator

    model = pipeline.return_model()
    train_dataset, eval_dataset = pipeline.return_dataset()
    data_collator = pipeline.return_collator()
    processor = pipeline.return_processor()
    processor.save_pretrained(processor_dir)
    _print("[i] Model type:", type(model))

    # deepspeed config
    if config.training.num_gpus > 1 and not config.training.use_ddp:
        deepspeed_config = config.get_deepspeed_config()
    else:
        deepspeed_config = None

    # For now we will let batch_size override global_batch_size, in future we will deprecate batch_size
    if config.training.batch_size is None:
        per_device_train_batch_size = config.training.global_batch_size // config.training.num_gpus
    else:
        per_device_train_batch_size = config.training.batch_size

    print(f"per_device_train_batch_size: {per_device_train_batch_size}")

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=config.training.max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=config.training.eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler_type,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        max_grad_norm=config.training.max_grad_norm,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        tf32=config.training.tf32,
        gradient_checkpointing=config.training.gradient_checkpointing,
        optim=config.training.optim,
        dataloader_num_workers=config.training.dataloader_num_workers,
        report_to="wandb" if config.training.use_wandb else "none",
        seed=config.data.seed,
        deepspeed=deepspeed_config,
        ddp_find_unused_parameters=False,
        ddp_timeout=72000,
        ddp_bucket_cap_mb=config.training.ddp_bucket_cap_mb,
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        batch_eval_metrics=True,
        remove_unused_columns=config.training.remove_unused_columns,
        ignore_data_skip=True,
    )

    # Create trainer
    trainer = RLDXTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        multiprocessing_context=config.data.multiprocessing_context,
    )

    trainer.add_callback(
        CheckpointFormatCallback(
            run_name=experiment_name,
            exp_cfg_dir=save_cfg_dir,
            processor_dir=processor_dir,
        )
    )

    if config.training.save_best_eval_metric_name != "":
        trainer.add_callback(
            BestMetricCheckpointCallback(
                metric_name=config.training.save_best_eval_metric_name,
                greater_is_better=config.training.save_best_eval_metric_greater_is_better,
                exp_cfg_dir=save_cfg_dir,
            )
        )

    new_param_warmup_steps = getattr(config.training, "new_param_warmup_steps", 0)
    new_param_names = getattr(model, "_new_param_names", set())
    if new_param_warmup_steps > 0 and new_param_names:
        trainer.add_callback(
            NewParamWarmupCallback(
                warmup_steps=new_param_warmup_steps,
                new_param_names=new_param_names,
                trainer=trainer,
            )
        )
        _print(
            f"[i] NewParamWarmupCallback registered: {new_param_warmup_steps} steps, {len(new_param_names)} new params"
        )
    elif new_param_warmup_steps > 0 and not new_param_names:
        _print(
            "[w] --new-param-warmup-steps set but no new parameters detected (no checkpoint or no missing keys). Skipping warmup."
        )

    if hasattr(train_dataset, "get_initial_actions"):
        initial_actions = train_dataset.get_initial_actions()
        if initial_actions:
            initial_actions_path = save_cfg_dir / INITIAL_ACTIONS_FILENAME
            save_initial_actions(initial_actions, initial_actions_path)
            _print(f"Saved {len(initial_actions)} initial actions to {initial_actions_path}")

    # Train
    _print("🚀 Starting training...")
    if config.training.enable_profiling:
        from functools import partial

        _print(f"{global_rank} Starting training with profiling...")

        def on_trace_ready_handler(trainer, profile_dir, prof):
            output_path = (
                profile_dir / f"trace_rank_{global_rank}_iter_{trainer.state.global_step}.json"
            )
            prof.export_chrome_trace(str(output_path))
            _print(f"Trace saved to {output_path}")

        profile_dir = output_dir / "profiling"
        profile_dir.mkdir(parents=True, exist_ok=True)

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=3, repeat=1),
            # profile_memory=True,
            with_stack=True,
            # record_shapes=True,
            on_trace_ready=partial(on_trace_ready_handler, trainer, profile_dir),
        ) as prof:
            trainer.add_callback(ProfCallback(prof=prof))
            trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train(resume_from_checkpoint=True)  # Resume from checkpoint if available

    # Save final model
    trainer.save_model()
    _print(f"Model saved to {output_dir}")

    if config.training.assert_loss_less_than is not None:
        final_loss = trainer.loss
        if final_loss.item() > config.training.assert_loss_less_than:
            raise AssertionError(
                f"Loss too high: {final_loss.item()} vs {config.training.assert_loss_less_than})"
            )

    # Cleanup
    if hasattr(train_dataset, "close"):
        train_dataset.close()
    if eval_dataset is not None and hasattr(eval_dataset, "close"):
        eval_dataset.close()
    _print("🚀 Training completed!")
