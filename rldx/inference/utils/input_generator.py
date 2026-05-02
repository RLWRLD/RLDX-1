"""VLM-specific benchmark utilities: synthetic input generation and parameter counting.

Shared utilities (PowerMonitor, measure_times) live in inference/utils/ and are
re-exported via setup/__init__.py.
"""

from __future__ import annotations

import torch


# Input generation

_DEFAULT_VLA_PROMPT = (
    "pick up the green cup from the counter and place it on the top shelf "
    "of the cabinet next to the blue plate then close the cabinet door carefully"
)


def generate_synthetic_input(
    model_path,
    num_images,
    image_height,
    image_width,
    concat_frames,
    prompt_tokens,
    device,
    seed=42,
    custom_prompt=None,
    num_frames=1,
):
    """Generate VL input using AutoProcessor.

    Args:
        num_images: Number of camera views.
        image_height, image_width: Image H and W in pixels.
        num_frames: Number of temporal frames per view. When > 1, produces
            `num_images * num_frames` images in time-major order
            (t0_v0, t0_v1, ..., t1_v0, ...), matching
            `processing_rldx.py`.
        prompt_tokens: Synthetic-token count; ignored when custom_prompt is set.
            If both are None, _DEFAULT_VLA_PROMPT is used.
        custom_prompt: Explicit text prompt string. Overrides prompt_tokens.
    """
    from PIL import Image
    from transformers import AutoProcessor

    torch.manual_seed(seed)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    # ``RLDXProcessor`` wraps an inner Qwen3-VL processor on
    # ``self.processor``.  The inference benchmarks just need the raw
    # tokenizer / image-processor surface (``apply_chat_template`` +
    # ``__call__(text=, images=)``), which the wrapper itself doesn't
    # expose — its ``__call__`` takes a list of training features.
    # Fall through to the inner processor when one is present.
    inner_processor = getattr(processor, "processor", processor)

    if concat_frames and num_images > 1:
        if num_frames > 1:
            raise ValueError("--concat-frames is incompatible with num_frames>1")
        images = [
            Image.new(
                "RGB",
                (image_width * num_images, image_height),
                color=(128, 128, 128),
            )
        ]
        actual_num_images = 1
    else:
        # Time-major order: T outer, V inner (matches processing_rldx.py).
        # PIL.Image.new takes (W, H).
        total = num_images * num_frames
        images = [
            Image.new("RGB", (image_width, image_height), color=(128, 128, 128))
            for _ in range(total)
        ]
        actual_num_images = num_images

    # Build prompt text
    if custom_prompt is not None:
        prompt_text = custom_prompt
    elif prompt_tokens is not None:
        # Legacy: synthetic tokens for backward compat
        prompt_text = "Hello" if prompt_tokens <= 1 else " ".join(["hello"] * prompt_tokens)
    else:
        prompt_text = _DEFAULT_VLA_PROMPT

    image_entries = [{"type": "image", "image": img} for img in images]
    messages = [
        {"role": "user", "content": image_entries + [{"type": "text", "text": prompt_text}]}
    ]

    text = inner_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = inner_processor(text=[text], images=images, return_tensors="pt", padding=True)

    vl_data = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            vl_data[k] = v.to(device)

    # Keys expected by vanilla backbone but not produced by processor
    if "image_wise_encoding" not in vl_data:
        vl_data["image_wise_encoding"] = torch.tensor([1], device=device)
    if "num_views" not in vl_data:
        vl_data["num_views"] = torch.tensor([actual_num_images], device=device)
    if num_frames > 1 and "num_frames" not in vl_data:
        vl_data["num_frames"] = torch.tensor([num_frames], device=device)

    from transformers.feature_extraction_utils import BatchFeature

    vl_input = BatchFeature(data=vl_data)

    info = {
        "num_images": actual_num_images,
        "num_frames": num_frames,
        "concat_frames": concat_frames,
        "image_height": image_height,
        "image_width": image_width,
        "prompt_text": prompt_text,
    }
    if "input_ids" in vl_data:
        ids = vl_data["input_ids"]
        info["seq_len"] = ids.shape[1]
        image_token_id = inner_processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        info["vision_tokens"] = int((ids == image_token_id).sum().item())
        info["text_tokens"] = info["seq_len"] - info["vision_tokens"]
        info["prompt_tokens"] = info["text_tokens"]
    if "pixel_values" in vl_data:
        info["pixel_values_shape"] = list(vl_data["pixel_values"].shape)

    return vl_input, info


# Parameter counting


def count_parameters(backbone):
    total = sum(p.numel() for p in backbone.parameters())
    vision = sum(p.numel() for p in backbone.qwen_model.model.visual.parameters())
    llm = sum(p.numel() for p in backbone.qwen_model.model.language_model.parameters())
    proj = sum(p.numel() for p in backbone.qwen_linear.parameters())
    return {"total": total, "vision": vision, "llm": llm, "projection": proj}
