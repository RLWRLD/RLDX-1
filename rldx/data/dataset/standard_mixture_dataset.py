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
from torch.utils.data import Dataset

from rldx.data.dataset.sharded_mixture_dataset import merge_statistics
from rldx.data.dataset.standard_single_step_dataset import StandardSingleStepDataset
from rldx.data.interfaces import BaseProcessor


class StandardMixtureDataset(Dataset):
    """
    Map-style dataset that combines multiple StandardSingleStepDatasets.

    Unlike ShardedMixtureDataset (which is IterableDataset), this is a plain
    torch.utils.data.Dataset. Shuffling and distributed sampling are handled
    externally by the DataLoader / DistributedSampler in RLDXTrainer.

    The internal _all_steps list is sorted by (dataset_idx, episode_idx) for
    cache locality — each DataLoader worker receives a contiguous slice of
    indices, so consecutive accesses tend to come from the same episode.

    Args:
        datasets: List of StandardSingleStepDataset instances to combine
        weights: Mixing weights for each dataset (will be normalised)
        processor: Data processor applied to all datasets
        seed: Random seed used for weighted over/under-sampling at construction
        override_pretraining_statistics: Whether to override pretraining statistics
    """

    def __init__(
        self,
        datasets: list[StandardSingleStepDataset],
        weights: list[float],
        processor: BaseProcessor,
        seed: int = 42,
        override_pretraining_statistics: bool = False,
    ):
        self.datasets = datasets
        self.weights = weights
        self.seed = seed
        self.processor = processor
        self.override_pretraining_statistics = override_pretraining_statistics

        # Merge statistics across embodiments and set processor on each sub-dataset
        self.merge_statistics()

        # Build weighted, sorted step list
        self._all_steps: list[tuple[int, int]] = self._build_step_list(seed)

    # ------------------------------------------------------------------
    # Statistics + processor wiring
    # ------------------------------------------------------------------

    def merge_statistics(self) -> None:
        """Merge per-embodiment statistics and configure the processor."""
        all_stats_by_emb: dict[str, list] = {}
        weights_by_emb: dict[str, list[float]] = {}
        for ds, w in zip(self.datasets, self.weights):
            emb = getattr(ds, "embodiment_tag", None)
            if emb is None:
                continue
            emb = emb.value
            if emb not in all_stats_by_emb:
                all_stats_by_emb[emb] = []
                weights_by_emb[emb] = []
            all_stats_by_emb[emb].append(ds.get_dataset_statistics())
            weights_by_emb[emb].append(w)

        stats_by_emb: dict[str, dict] = {}
        for emb, stats_list in all_stats_by_emb.items():
            stats_by_emb[emb] = {}
            # Collect all modality keys present across datasets (state, action, tactile, torque, etc.)
            all_modalities = set()
            for s in stats_list:
                all_modalities.update(s.keys())
            for modality in all_modalities:
                if modality in stats_list[0]:
                    modality_stats = [s[modality] for s in stats_list if modality in s]
                    stats_by_emb[emb][modality] = merge_statistics(
                        per_dataset_stats=modality_stats,
                        dataset_sampling_weights=[
                            w for w, s in zip(weights_by_emb[emb], stats_list) if modality in s
                        ],
                        is_relative_stats=(modality == "relative_action"),
                    )

        self.global_stats = stats_by_emb
        self.processor.set_statistics(
            self.global_stats, override=self.override_pretraining_statistics
        )
        for ds in self.datasets:
            ds.set_processor(self.processor)

    def get_dataset_statistics(self) -> dict:
        return self.global_stats

    # ------------------------------------------------------------------
    # Step-list construction
    # ------------------------------------------------------------------

    def _build_step_list(self, seed: int) -> list[tuple[int, int]]:
        """
        Build a flat weighted list of (dataset_idx, local_step_idx) pairs,
        sorted by (dataset_idx, episode_idx) for cache locality.

        Weights are applied by over- or under-sampling each dataset:
          total = max(len(ds) / w  for each normalised weight w)
          target_n[i] = round(total * normalised_weights[i])
        """
        rng = np.random.default_rng(seed)
        weights = np.array(self.weights, dtype=float)
        normalized_weights = weights / weights.sum()

        total = int(max(len(ds) / w for ds, w in zip(self.datasets, normalized_weights)))

        all_steps: list[tuple[int, int]] = []
        for ds_idx, (ds, w) in enumerate(zip(self.datasets, normalized_weights)):
            target_n = max(1, round(total * w))
            ds_len = len(ds)
            if target_n <= ds_len:
                chosen = rng.choice(ds_len, size=target_n, replace=False)
            else:
                chosen = rng.choice(ds_len, size=target_n, replace=True)
            all_steps.extend((ds_idx, int(i)) for i in chosen)

        # Sort by (ds_idx, ep_idx) for cache locality
        def _sort_key(item: tuple[int, int]) -> tuple[int, int]:
            ds_idx, local_idx = item
            ep_idx, _ = self.datasets[ds_idx].all_steps[local_idx]
            return (ds_idx, ep_idx)

        all_steps.sort(key=_sort_key)

        print(
            f"StandardMixtureDataset: {len(all_steps)} steps from {len(self.datasets)} dataset(s)"
        )
        return all_steps

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._all_steps)

    def __getitem__(self, idx: int) -> dict:
        ds_idx, local_step_idx = self._all_steps[idx]
        return self.datasets[ds_idx][local_step_idx]

    # ------------------------------------------------------------------
    # reset_seed: called by RLDXTrainer on checkpoint resume
    # ------------------------------------------------------------------

    def reset_seed(self, seed: int) -> None:
        """Rebuild the step list with a new seed (called on checkpoint resume)."""
        self.seed = seed
        self._all_steps = self._build_step_list(seed)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_initial_actions(self) -> list:
        initial_actions: list = []
        for ds in self.datasets:
            if hasattr(ds, "get_initial_actions"):
                initial_actions.extend(ds.get_initial_actions())
        return initial_actions
