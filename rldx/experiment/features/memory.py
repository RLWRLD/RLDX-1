"""Memory feature — HAMLET memory-augmented cognition tokens."""

from __future__ import annotations

from rldx.experiment.features.context import AssemblyContext
from rldx.utils.dist import rank_zero_print as _print


class MemoryFeature:
    name = "memory"
    requires = frozenset()

    @staticmethod
    def is_active(cli) -> bool:
        return cli.use_memory

    @staticmethod
    def apply(ctx: AssemblyContext) -> None:
        cli = ctx.cli
        model = ctx.model

        model.use_memory = True
        model.memory_length = cli.memory_length
        model.memory_stride = cli.memory_stride
        model.memory_n_cog_tokens = cli.memory_n_cog_tokens
        model.concat_memory = cli.concat_memory
        model.memory_dropout_prob = cli.memory_dropout_prob
        if cli.memory_dropout_prob > 0.0:
            assert cli.concat_memory, "memory_dropout_prob > 0.0 requires concat_memory=True"
        if cli.blockwise_attn_for_memory:
            model.memory_cfg["use_causal_attn"] = False

        stride = cli.memory_stride
        anchors = {-(cli.memory_length - 1 - i) * stride for i in range(cli.memory_length)}
        for emb_key in ctx.modality_configs:
            if "video" in ctx.modality_configs[emb_key]:
                ctx.add_anchors(emb_key, "video", anchors)
        ctx.data.allow_padding = True

        _print(
            f"[i] use_memory: True, memory_length: {cli.memory_length}, "
            f"memory_stride: {cli.memory_stride}, video_anchors: {sorted(anchors)}"
        )

    @staticmethod
    def required_state_dict_prefixes() -> frozenset[str]:
        # Placeholder for CheckpointShape (T-C) in a follow-up PR
        return frozenset()
