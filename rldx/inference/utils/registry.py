"""Model registry for the inference benchmarks.

Each entry pairs a checkpoint with the kwargs the loader passes through
to ``VTCQwen3VLBackbone``. The three production entries mirror the
RLDX-1 deployment lineup:

    rldx_1_pretrain         — base, video-only (no memory / physics)
    rldx_1_midtrain_allex   — video + motion + memory + torque
    rldx_1_midtrain_droid   — video + motion + memory + tactile + torque
"""

from __future__ import annotations


# TODO(release): replace ``TBD_RLDX_1_*`` with the published HF repo IDs.
# Local-path loading works without these — pass ``--model-path /local/dir``
# to the benchmark / server CLIs.
MODEL_REGISTRY = {
    "rldx_1_pretrain": {
        "hf_path": "TBD_RLDX_1_PRETRAIN",
        "processor_path": "TBD_RLDX_1_PRETRAIN",
        # Backbone-only — the rest of the action-model plumbing is
        # initialised fresh by the loader.
        "load_mode": "extract_backbone",
        "default_args": {
            "use_cog_tokens": True,
            "n_cog_tokens": 64,
            "select_layer": 18,
        },
    },
    "rldx_1_midtrain_allex": {
        "hf_path": "TBD_RLDX_1_MIDTRAIN_ALLEX",
        "processor_path": "TBD_RLDX_1_MIDTRAIN_ALLEX",
        # Carries memory + torque physics weights — load the full model.
        "load_mode": "full",
        "default_args": {
            "use_cog_tokens": True,
            "n_cog_tokens": 64,
            "select_layer": 18,
        },
    },
    "rldx_1_midtrain_droid": {
        "hf_path": "TBD_RLDX_1_MIDTRAIN_DROID",
        "processor_path": "TBD_RLDX_1_MIDTRAIN_DROID",
        # Same shape as ALLEX with tactile + torque physics.
        "load_mode": "full",
        "default_args": {
            "use_cog_tokens": True,
            "n_cog_tokens": 64,
            "select_layer": 18,
        },
    },
    # VTC backbone variant; loaded via the dynamic-import string.  Issue
    # #29 Phase 6 renamed the model_type from ``contextvla_qwen3_vl`` to
    # ``vtc_qwen3_vl``; the registry key follows.
    "vtc_qwen3_vl_8b": {
        "hf_path": "huiwon/roboalign_contextvla_oxe_sft_1epoch",
        "backbone_cls": "rldx.model.modules.backbone.adapter.VTCQwen3VLBackbone",
    },
}


def is_placeholder_path(hf_path) -> bool:
    """True for placeholder ``TBD_*`` strings; lets loaders fail loudly."""
    return isinstance(hf_path, str) and hf_path.startswith("TBD_")
