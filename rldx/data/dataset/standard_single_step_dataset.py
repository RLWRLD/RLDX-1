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

from pathlib import Path
from typing import Any

import pandas as pd
from torch.utils.data import Dataset

from rldx.data.interfaces import BaseProcessor
from rldx.data.types import EmbodimentTag, MessageType, ModalityConfig

from .lerobot_episode_loader import LeRobotEpisodeLoader
from .sharded_single_step_dataset import extract_step_data


class StandardSingleStepDataset(Dataset):
    """
    Map-style single-step dataset for VLA training.

    Unlike ShardedSingleStepDataset, this is a standard PyTorch map-style Dataset
    where each index directly maps to a (episode, step) pair. The HuggingFace
    Trainer handles distributed sampling via DistributedSampler, and this dataset
    is consumed by StandardMixtureDataset which handles its own shuffling/distribution.

    Key differences from ShardedSingleStepDataset:
    - map-style Dataset instead of the shard-based system
    - No pre-sharding; episodes are loaded on demand
    - Simple last-episode cache to amortize per-episode loading cost
    - No episode_sampling_rate — all valid steps are included

    The all_steps list is sorted by (ep_idx, step_idx) so that consecutive accesses
    within the same episode benefit from the single-episode cache.

    Args:
        dataset_path: Path to LeRobot format dataset directory
        embodiment_tag: Embodiment identifier for cross-embodiment training
        modality_configs: Configuration for each modality (sampling, keys)
        video_backend: Video decoding backend ('torchcodec', 'decord', etc.)
        video_backend_kwargs: Additional arguments for video backend
        allow_padding: Whether to allow padding of indices to valid range [0, max_length - 1]
    """

    def __init__(
        self,
        dataset_path: str | Path,
        embodiment_tag: EmbodimentTag,
        modality_configs: dict[str, ModalityConfig],
        video_backend: str = "torchcodec",
        video_backend_kwargs: dict[str, Any] | None = None,
        allow_padding: bool = False,
    ):
        self.dataset_path = Path(dataset_path)
        self.embodiment_tag = embodiment_tag
        self.modality_configs = modality_configs
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs
        self.allow_padding = allow_padding
        self.processor: BaseProcessor | None = None

        action_delta_indices = modality_configs["action"].delta_indices
        self.action_horizon = max(action_delta_indices) - min(action_delta_indices) + 1

        self.episode_loader = LeRobotEpisodeLoader(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            video_backend=video_backend,
            video_backend_kwargs=video_backend_kwargs,
        )

        # Build a flat, episode-sorted list of (ep_idx, step_idx) pairs
        self.all_steps: list[tuple[int, int]] = self._build_all_steps()

        # Last-episode cache: avoids re-loading a parquet/video for each step
        self._cached_ep_idx: int | None = None
        self._cached_ep_data: pd.DataFrame | None = None

        print(
            f"StandardSingleStepDataset: {len(self.all_steps)} steps "
            f"across {len(self.episode_loader)} episodes in {self.dataset_path}"
        )

    def _build_all_steps(self) -> list[tuple[int, int]]:
        """Build flat list of valid (ep_idx, step_idx) pairs sorted by ep_idx."""
        all_steps = []
        for ep_idx in range(len(self.episode_loader)):
            ep_len = self.episode_loader.get_episode_length(ep_idx)
            effective_len = max(0, ep_len - self.action_horizon + 1)
            for step_idx in range(effective_len):
                all_steps.append((ep_idx, step_idx))
        return all_steps

    def __len__(self) -> int:
        return len(self.all_steps)

    def set_processor(self, processor: BaseProcessor) -> None:
        self.processor = processor

    def get_dataset_statistics(self) -> dict:
        return self.episode_loader.get_dataset_statistics()

    def get_initial_actions(self):
        return self.episode_loader.get_initial_actions()

    def __getitem__(self, idx: int) -> dict:
        assert self.processor is not None, "Processor must be set before calling __getitem__"
        ep_idx, step_idx = self.all_steps[idx]

        # Load episode data, reusing the cache when possible
        if self._cached_ep_idx != ep_idx:
            self._cached_ep_data = self.episode_loader[ep_idx]
            self._cached_ep_idx = ep_idx

        assert self._cached_ep_data is not None
        vla_step_data = extract_step_data(
            self._cached_ep_data,
            step_idx,
            self.modality_configs,
            self.embodiment_tag,
            self.allow_padding,
        )
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        return self.processor(messages)
