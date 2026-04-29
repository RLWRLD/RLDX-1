"""motion module feature — Motion-Aware Spatio-Temporal Summarization (requires video)."""

from __future__ import annotations

from rldx.experiment.features.context import AssemblyContext
from rldx.utils.dist import rank_zero_print as _print


class MossFeature:
    name = "motion"
    requires = frozenset({"video"})  # T-E: enforced by _check_dependencies

    @staticmethod
    def is_active(cli) -> bool:
        return cli.use_motion

    @staticmethod
    def apply(ctx: AssemblyContext) -> None:
        cli = ctx.cli
        model = ctx.model

        model.use_motion = True
        model.motion_insert_layer = cli.motion_insert_layer
        model.motion_injection_point = cli.motion_injection_point
        model.motion_pool_type = cli.motion_pool_type
        model.motion_drop = cli.motion_drop
        model.motion_gradient_check = cli.motion_gradient_check
        _print(
            f"[i] use_motion: True, motion_insert_layer: {cli.motion_insert_layer}, "
            f"motion_injection_point: {cli.motion_injection_point}, "
            f"motion_pool_type: {cli.motion_pool_type}, motion_drop: {cli.motion_drop}"
        )

    @staticmethod
    def required_state_dict_prefixes() -> frozenset[str]:
        return frozenset()
