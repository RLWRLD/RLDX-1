"""Physics feature — tactile/torque conditioning stream in the action head."""

from __future__ import annotations

from rldx.experiment.features.context import AssemblyContext
from rldx.utils.dist import rank_zero_print as _print


class PhysicsFeature:
    name = "physics"
    requires = frozenset()

    @staticmethod
    def is_active(cli) -> bool:
        return cli.use_physics

    @staticmethod
    def apply(ctx: AssemblyContext) -> None:
        cli = ctx.cli
        model = ctx.model
        modality_configs = ctx.modality_configs

        physics_keys = cli.physics_keys or []
        physics_dims = cli.physics_dims or []
        assert len(physics_keys) > 0, (
            "--use-physics requires --physics-keys with at least one key (e.g. tactile, torque)"
        )
        assert len(physics_dims) == len(physics_keys), (
            f"--physics-dims length ({len(physics_dims)}) must match "
            f"--physics-keys length ({len(physics_keys)})"
        )
        # Read-only validation: shared delta_indices across physics modalities
        for emb_key in modality_configs:
            ref_delta = None
            has_any_physics = any(pk in modality_configs[emb_key] for pk in physics_keys)
            if not has_any_physics:
                if not cli.allow_missing_physics:
                    raise ValueError(
                        f"Embodiment '{emb_key}' has no physics modality keys "
                        f"{physics_keys} in its modality config. "
                        f"Use --allow-missing-physics to allow this."
                    )
                continue
            for pk in physics_keys:
                if pk in modality_configs[emb_key]:
                    di = modality_configs[emb_key][pk].delta_indices
                    if ref_delta is None:
                        ref_delta = di
                    else:
                        assert di == ref_delta, (
                            f"All physics modalities must share the same "
                            f"delta_indices. Got mismatch for '{pk}': {di} vs "
                            f"reference: {ref_delta}"
                        )

        model.use_physics = True
        model.physics_keys = physics_keys
        model.physics_dims = physics_dims
        model.physics_loss_weight = cli.physics_loss_weight
        model.allow_missing_physics = cli.allow_missing_physics
        model.physics_dropout_prob = cli.physics_dropout_prob
        _print(
            f"[i] use_physics: True, physics_keys: {physics_keys}, "
            f"physics_dims: {physics_dims}, physics_dim (total): {sum(physics_dims)}, "
            f"loss_weight: {cli.physics_loss_weight}, "
            f"allow_missing: {cli.allow_missing_physics}, "
            f"dropout_prob: {cli.physics_dropout_prob}"
        )

    @staticmethod
    def required_state_dict_prefixes() -> frozenset[str]:
        return frozenset()
