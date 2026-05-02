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

import logging
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
from termcolor import colored
import tyro

from rldx.configs.base_config import Config, get_default_config
from rldx.configs.train_config import TrainConfig
from rldx.experiment.assembly import (
    AssemblyInputs,
    _assert_dataset_config_mutually_exclusive,
    assemble_run_config,
    build_dataset_specs,
    build_pt_dataset_specs,
)
from rldx.experiment.experiment import run
from rldx.experiment.utils import load_modality_config, snapshot_model_config
from rldx.utils.dist import rank_zero_print as _print


def _resolve_base_model(
    base_path_or_id: Optional[str], revision: Optional[str] = None
) -> Optional[str]:
    """Return a local path for the base checkpoint, downloading via HF hub if needed.

    Args:
        base_path_or_id: HF Hub repo id or local path. ``None`` → no base model.
        revision: HF git commit / branch / tag to pin. Issue #22 — previously
            this call ignored ``--model-revision`` entirely and pulled HEAD,
            silently breaking paper-reproduction runs.

    Returns ``None`` when no base model was requested.
    """
    if not base_path_or_id:
        return None
    if "/" in base_path_or_id and not Path(base_path_or_id).exists():
        return snapshot_download(
            repo_id=base_path_or_id,
            repo_type="model",
            local_dir=None,
            revision=revision,
        )
    return base_path_or_id


def _select_datasets_and_tag(cli_config: TrainConfig):
    """Branch on pretrain vs finetune mode and build the dataset specs
    (and, for finetune, resolve the embodiment tag via the fail-fast path).

    Returns `(datasets, embodiment_tag_or_None)`. The `__main__` block
    calls this directly so a reviewer-driven fault injection on the
    finetune branch (e.g. reverting the call to `.embodiment_tag.value`)
    is observable from unit tests — i.e., the actual call site is pinned,
    not just the resolver in isolation.
    """
    _assert_dataset_config_mutually_exclusive(cli_config)

    use_pt = (cli_config.pt_dataset_root is not None) and (cli_config.pt_dataset_mix is not None)
    if use_pt:
        return build_pt_dataset_specs(cli_config), None

    embodiment_tag = _resolve_finetune_embodiment_tag(cli_config)
    return build_dataset_specs(cli_config, embodiment_tag), embodiment_tag


def _resolve_finetune_embodiment_tag(cli_config: TrainConfig) -> str:
    """Return the finetune-mode embodiment tag string, fail-fast on None.

    PR #15 review (Gerald, 2026-04-22) — the previous flow was
    `cli_config.embodiment_tag.value` with no None guard, which crashed
    with `AttributeError` and a stack trace pointing into tyro internals.
    The original docstring promised "Defaults to GENERAL_EMBODIMENT when
    None" but no code path implemented that default. We chose Option A
    (fail-fast over silent default) so a user picking the wrong dataset
    can't silently inherit a wrong embodiment.
    """
    if cli_config.embodiment_tag is None:
        raise ValueError(
            "--embodiment-tag is required for finetune mode "
            "(--dataset-path / --dataset-paths). Pass it explicitly, e.g. "
            "--embodiment-tag GENERAL_EMBODIMENT. Pretrain mode "
            "(--pt-dataset-root / --pt-dataset-mix) does not need this flag."
        )
    return cli_config.embodiment_tag.value


def _load_yaml_config(base_model_path: Optional[str]) -> Optional[Config]:
    """Load experiment_cfg/config.yaml shipped with the checkpoint, if present.

    Returns None when the file is missing so the caller falls back to
    CLI/default config.
    """
    if base_model_path is None:
        return None
    config_yaml = Path(f"{base_model_path}/experiment_cfg/config.yaml")
    if not config_yaml.exists():
        logging.warning(
            f"experiment_cfg/config.yaml not found under {base_model_path}. "
            "Falling back to CLI/default config."
        )
        return None
    run_config = get_default_config().load(config_yaml)
    _print(f"[i] Loaded config from {config_yaml}")
    return run_config


if __name__ == "__main__":
    if "LOG_LEVEL" not in os.environ:
        os.environ["LOG_LEVEL"] = "INFO"
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, level))

    cli_config = tyro.cli(TrainConfig, description=__doc__)
    datasets, _ = _select_datasets_and_tag(cli_config)

    # All rank workers should register for the modality config
    if cli_config.modality_config_path is not None:
        load_modality_config(cli_config.modality_config_path)

    # I/O boundary: HF download + YAML read. Isolated here so assemble_run_config
    # remains pure.
    base_model_path = _resolve_base_model(
        cli_config.base_model_path, revision=cli_config.model_revision
    )
    loaded_yaml_config = _load_yaml_config(base_model_path)
    loaded_ckpt_model_snapshot = (
        snapshot_model_config(loaded_yaml_config.model) if loaded_yaml_config is not None else None
    )

    # Pure assembly (model + features + training overrides)
    run_config = assemble_run_config(
        AssemblyInputs(
            cli=cli_config,
            datasets=datasets,
            base_model_path=base_model_path,
            loaded_yaml_config=loaded_yaml_config,
            loaded_ckpt_model_snapshot=loaded_ckpt_model_snapshot,
        )
    )

    _print(colored(f"[i] {cli_config=}", "yellow"))
    run(run_config)
