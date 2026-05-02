"""Per-call shared state for `assemble_run_config`. Feature modules push
anchors and strides per (embodiment, modality); `finalize()` writes back

    delta_indices = sorted({a + s for a in anchors for s in strides})

with anchors seeded from existing delta_indices and strides seeded with {0}.
Set union makes feature application order irrelevant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from rldx.configs.base_config import Config
from rldx.configs.train_config import TrainConfig


@dataclass
class AssemblyContext:
    cli: TrainConfig
    run_config: Config
    _anchors: dict[tuple[str, str], set[int]] = field(default_factory=dict)
    _strides: dict[tuple[str, str], set[int]] = field(default_factory=dict)
    _touched: set[tuple[str, str]] = field(default_factory=set)

    @property
    def model(self):
        return self.run_config.model

    @property
    def data(self):
        return self.run_config.data

    @property
    def modality_configs(self):
        return self.run_config.data.modality_configs

    def add_anchors(self, emb_key: str, modality: str, anchors: Iterable[int]) -> None:
        """Union `anchors` into this (emb, modality)'s anchor set.

        First call initializes from existing delta_indices."""
        k = (emb_key, modality)
        self._touched.add(k)
        if k not in self._anchors:
            existing = self.modality_configs[emb_key][modality].delta_indices
            self._anchors[k] = set(existing)
        self._anchors[k] |= set(anchors)

    def add_strides(self, emb_key: str, modality: str, strides: Iterable[int]) -> None:
        """Union `strides` into this (emb, modality)'s stride set.

        First call initializes to {0}."""
        k = (emb_key, modality)
        self._touched.add(k)
        if k not in self._strides:
            self._strides[k] = {0}
        self._strides[k] |= set(strides)

    def finalize(self) -> None:
        """Write composed delta_indices back. Untouched modalities stay as-is."""
        for emb_key, mod_key in self._touched:
            anchors = self._anchors.get((emb_key, mod_key))
            if anchors is None:
                anchors = set(self.modality_configs[emb_key][mod_key].delta_indices)
            strides = self._strides.get((emb_key, mod_key), {0})
            delta = sorted({a + s for a in anchors for s in strides})
            self.modality_configs[emb_key][mod_key].delta_indices = delta
