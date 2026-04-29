"""FeatureModule registry. Each module owns one optional feature's
CLI→config mutation (Memory / Video / motion module / Physics). Composition order
across features is irrelevant because `AssemblyContext` unions anchors and
strides as sets.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rldx.configs.train_config import TrainConfig
from rldx.experiment.features.context import AssemblyContext
from rldx.experiment.features.memory import MemoryFeature
from rldx.experiment.features.motion import MossFeature
from rldx.experiment.features.physics import PhysicsFeature
from rldx.experiment.features.video import VideoFeature


@runtime_checkable
class FeatureModule(Protocol):
    """Contract for one optional feature."""

    name: str
    """Stable identifier used by _check_dependencies for cross-feature refs."""

    requires: frozenset[str]
    """Other feature names that must be active simultaneously."""

    @staticmethod
    def is_active(cli: TrainConfig) -> bool:
        """Does this CLI activate this feature?"""

    @staticmethod
    def apply(ctx: AssemblyContext) -> None:
        """Mutate ctx.run_config directly for owned model/data fields;
        contribute to modality via ctx.add_anchors / ctx.add_strides."""

    @staticmethod
    def required_state_dict_prefixes() -> frozenset[str]:
        """State_dict key prefixes this feature contributes to a checkpoint.
        Used by CheckpointShape verification (spec §3.5; follow-up PR)."""


FEATURES: tuple[type[FeatureModule], ...] = (
    MemoryFeature,
    VideoFeature,
    MossFeature,
    PhysicsFeature,
)


def _check_dependencies(active: tuple[type[FeatureModule], ...]) -> None:
    """Enforce declared `requires` before any F.apply() runs.

    Raises ValueError listing all unmet deps across all active features.
    """
    active_names = {F.name for F in active}
    errors = []
    for F in active:
        missing = F.requires - active_names
        if missing:
            errors.append(f"feature '{F.name}' requires {sorted(missing)}, not active")
    if errors:
        raise ValueError("; ".join(errors))


__all__ = [
    "FeatureModule",
    "FEATURES",
    "MemoryFeature",
    "VideoFeature",
    "MossFeature",
    "PhysicsFeature",
    "AssemblyContext",
    "_check_dependencies",
]
