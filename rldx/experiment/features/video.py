"""Video feature — VTC video-frame input."""

from __future__ import annotations

from rldx.experiment.features.context import AssemblyContext
from rldx.utils.dist import rank_zero_print as _print


class VideoFeature:
    name = "video"
    requires = frozenset()

    @staticmethod
    def is_active(cli) -> bool:
        # Release codebase always uses the VTC backbone (Issue #36 / PR #39
        # removed the vanilla Qwen3 path, Issue #53 dropped ``--use-video``).
        # Video tokens are now an architectural invariant rather than an
        # ablation knob.
        return True

    @staticmethod
    def apply(ctx: AssemblyContext) -> None:
        cli = ctx.cli
        model = ctx.model

        model.use_video = True
        model.video_length = cli.video_length
        model.video_stride = cli.video_stride

        _print(f"[i] use_video: True, video_length: {cli.video_length}")

        stride = cli.video_stride
        strides = {(i - (cli.video_length - 1)) * stride for i in range(cli.video_length)}
        for emb_key in ctx.modality_configs:
            if "video" in ctx.modality_configs[emb_key]:
                ctx.add_strides(emb_key, "video", strides)

        ctx.data.allow_padding = True
        _print(f"[i] Using video strides: {sorted(strides)}")

    @staticmethod
    def required_state_dict_prefixes() -> frozenset[str]:
        return frozenset()
