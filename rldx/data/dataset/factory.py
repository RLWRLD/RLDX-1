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

import numpy as np
import torch
from tqdm import tqdm

from rldx.configs.base_config import Config
from rldx.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
from rldx.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from rldx.data.dataset.standard_mixture_dataset import StandardMixtureDataset
from rldx.data.dataset.standard_single_step_dataset import StandardSingleStepDataset
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.interfaces import BaseProcessor
from rldx.data.stats import generate_rel_stats, generate_stats
from rldx.experiment.dist_utils import barrier


class DatasetFactory:
    """
    Factory class for building training datasets. Model-agnostic.

    Supports two dataset modes (configured via ``config.data.dataset_mode``):

    * ``"sharded"`` (default): Pre-shards episodes into fixed-size shard files.
      Episodes are grouped by shard, loaded in bulk, and background-prefetched.
      Controlled by ``shard_size`` and ``episode_sampling_rate``.

    * ``"standard"``: Map-style dataset where every valid (episode, step) pair
      is a directly addressable index.  No pre-sharding; the per-episode cache
      inside StandardSingleStepDataset amortises episode loading cost.
      ``episode_sampling_rate`` is ignored; all valid steps are included.
    """

    def __init__(self, config: Config):
        self.config = config

    def _ensure_stats(self, dataset_path: str, embodiment_tag: str) -> None:
        """Generate dataset statistics if not already present (rank-0 only, then barrier)."""
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                generate_stats(dataset_path)
                generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
        else:
            generate_stats(dataset_path)
            generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
        barrier()

    def build(
        self, processor: BaseProcessor
    ) -> tuple[ShardedMixtureDataset | StandardMixtureDataset, None]:
        """Build the dataset. Returns a tuple of (train_dataset, eval_dataset)."""
        assert self.config.training.eval_strategy == "no", (
            "Sharded dataset does not support evaluation sets"
        )

        dataset_mode = getattr(self.config.data, "dataset_mode", "sharded")
        assert dataset_mode in ("sharded", "standard"), (
            f"Unknown dataset_mode '{dataset_mode}'. Choose 'sharded' or 'standard'."
        )

        all_datasets = []
        all_weights = []
        for dataset_spec in tqdm(
            self.config.data.datasets,
            total=len(self.config.data.datasets),
            desc="Initializing datasets",
        ):
            datasets = []
            for dataset_path in dataset_spec.dataset_paths:
                embodiment_tag = dataset_spec.embodiment_tag
                assert embodiment_tag is not None, "Embodiment tag is required"
                assert self.config.data.mode == "single_turn", "Only single turn mode is supported"

                self._ensure_stats(dataset_path, embodiment_tag)

                if dataset_mode == "sharded":
                    dataset = ShardedSingleStepDataset(
                        dataset_path=dataset_path,
                        embodiment_tag=EmbodimentTag(embodiment_tag),
                        modality_configs=self.config.data.modality_configs[embodiment_tag],
                        video_backend=self.config.data.video_backend,
                        shard_size=self.config.data.shard_size,
                        episode_sampling_rate=self.config.data.episode_sampling_rate,
                        seed=self.config.data.seed,
                        allow_padding=self.config.data.allow_padding,
                    )
                else:
                    dataset = StandardSingleStepDataset(
                        dataset_path=dataset_path,
                        embodiment_tag=EmbodimentTag(embodiment_tag),
                        modality_configs=self.config.data.modality_configs[embodiment_tag],
                        video_backend=self.config.data.video_backend,
                        allow_padding=self.config.data.allow_padding,
                    )

                datasets.append(dataset)

            dataset_lengths = np.array([len(dataset) for dataset in datasets])
            dataset_relative_lengths = dataset_lengths / dataset_lengths.sum()
            for dataset, relative_length in zip(datasets, dataset_relative_lengths):
                weight = relative_length * dataset_spec.mix_ratio
                all_datasets.append(dataset)
                all_weights.append(weight)

        if dataset_mode == "sharded":
            train_dataset = ShardedMixtureDataset(
                datasets=all_datasets,
                weights=all_weights,
                processor=processor,
                seed=self.config.data.seed,
                training=True,
                num_shards_per_epoch=self.config.data.num_shards_per_epoch,
                override_pretraining_statistics=self.config.data.override_pretraining_statistics,
            )
        else:
            train_dataset = StandardMixtureDataset(
                datasets=all_datasets,
                weights=all_weights,
                processor=processor,
                seed=self.config.data.seed,
                override_pretraining_statistics=self.config.data.override_pretraining_statistics,
            )

        return train_dataset, None
