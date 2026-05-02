"""Minimal backbone patch for GraphSafeQwen3VLBackbone integration.

Single patch: backbone.forward → gs_backbone.forward + BatchFeature wrapping.
"""

from __future__ import annotations

import types

from transformers.feature_extraction_utils import BatchFeature


def patch_backbone(backbone, gs_backbone):
    """Patch backbone.forward to delegate to the unified GraphSafe VLM.

    Args:
        backbone: ContextVLAQwen3VLBackbone (already on device)
        gs_backbone:   GraphSafeQwen3VLBackbone instance
    """
    image_token_id = backbone.qwen_model.model.config.image_token_id

    def _forward(self, vl_input):
        self.set_frozen_modules_to_eval_mode()
        features = gs_backbone(vl_input)
        return BatchFeature(
            data={
                "backbone_features": features,
                "backbone_attention_mask": vl_input.get("attention_mask"),
                "image_mask": (vl_input["input_ids"] == image_token_id),
            }
        )

    backbone.forward = types.MethodType(_forward, backbone)
