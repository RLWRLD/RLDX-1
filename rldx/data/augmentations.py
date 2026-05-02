"""Image augmentation pipeline for RLDX.

Pipeline (applied to every camera view):

  1. AspectAreaResizeAndCrop
       Aspect-ratio-preserving resize so the total pixel area is at most
       ``image_max_area``, then center-crop both spatial dims down to
       multiples of ``image_resize_m``.
       Deterministic — used by both train and eval.

  2. (optional) Fractional crop + resize-back
       Crop a ``random_crop_fraction`` sub-region and resize it back to the
       post-step-1 shape, so downstream stages always see a fixed output size.
       - Training: random position (FractionalRandomCropAndResize).
       - Inference: center position (FractionalCenterCropAndResize).
       Skipped entirely when ``random_crop_fraction`` is None (no-op).

  3. (train only, optional) Rotate / ColorJitter
       Standard photometric & mild geometric augmentation.

The builder :func:`build_image_transformations_albumentations` returns
``(train_transform, eval_transform)``. The processor stores both and selects
between them based on :pyattr:`ProcessorMixin.training` (PyTorch-style flag).
"""

from __future__ import annotations

import math
import warnings

import albumentations as A
import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Geometry helper (pure function — unit-tested for exact values)
# ---------------------------------------------------------------------------


def resize_preserve_aspect_area_then_crop(
    h: int, w: int, max_area: int = 256**2, m: int = 32
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Compute aspect-ratio-preserving resize + m-aligned crop sizes.

    The resize never upscales. It chooses the largest scale such that the
    resulting area is at most ``max_area`` **and** the shorter side is a
    multiple of ``m``. The longer side then follows from the aspect ratio,
    and is finally cropped (not resized) to a multiple of ``m``.

    Args:
        h: Input height.
        w: Input width.
        max_area: Maximum pixel area budget for the resized image.
        m: Alignment multiple for the output dimensions.

    Returns:
        ``((h_r, w_r), (h_c, w_c))`` where the first pair is the resize
        target (aspect ratio preserved) and the second pair is the final
        crop (both dims multiples of ``m``).
    """
    smax = min(1.0, math.sqrt(max_area / (h * w)))
    short, long_ = (h, w) if h <= w else (w, h)

    # largest ``m``-multiple short side that respects the area budget
    short_r = max(m, int((short * smax) // m) * m)
    s = short_r / short

    # aspect-preserving long side (floor so resized area <= max_area)
    long_r = int(long_ * s)

    h_r, w_r = (short_r, long_r) if h <= w else (long_r, short_r)
    h_c = h_r - (h_r % m)
    w_c = w_r - (w_r % m)
    return (h_r, w_r), (h_c, w_c)


# ---------------------------------------------------------------------------
# Replay helper (consistent stochastic transforms across multiple views)
# ---------------------------------------------------------------------------


def apply_with_replay(transform, images, replay=None):
    """Apply an albumentations transform to multiple images with replay.

    When ``transform`` is a :class:`A.ReplayCompose`, the first image produces
    replay data that is reused for subsequent images so that random params
    (rotation, jitter, crop origin, ...) are consistent across all views.

    Args:
        transform: ``A.Compose`` or ``A.ReplayCompose``.
        images: Iterable of PIL images (or numpy arrays).
        replay: Optional replay blob from an earlier call. When None, the
            first image creates fresh replay data.

    Returns:
        ``(tensors, replay)`` — a list of ``uint8`` tensors ``(C, H, W)`` and
        the replay blob (or None for plain ``A.Compose``).
    """
    transformed_tensors = []
    current_replay = replay
    has_replay = hasattr(transform, "replay")

    for img in images:
        if has_replay:
            if current_replay is None:
                augmented = transform(image=np.array(img))
                current_replay = augmented["replay"]
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    augmented = transform.replay(
                        image=np.array(img), saved_augmentations=current_replay
                    )
            img_array = augmented["image"]
        else:
            augmented = transform(image=np.array(img))
            img_array = augmented["image"]

        if img_array.dtype == np.float32:
            img_array = (img_array * 255).astype(np.uint8)
        elif img_array.dtype != np.uint8:
            raise ValueError(f"Unexpected data type: {img_array.dtype}")

        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        transformed_tensors.append(img_tensor)

    return transformed_tensors, current_replay


# ---------------------------------------------------------------------------
# Step 1 — aspect-area resize + m-aligned crop
# ---------------------------------------------------------------------------


class AspectAreaResizeAndCrop(A.DualTransform):
    """Resize preserving aspect ratio (area-constrained) then center-crop.

    Step 1.1: Resize so total area <= ``max_area``, short side aligned to
              ``m``, aspect ratio preserved.
    Step 1.2: Center-crop both dims to multiples of ``m`` (removes at most
              ``m-1`` pixels per dim).

    This is the first (and only required) stage of the image pipeline.
    """

    _logged_shapes: set = set()

    def __init__(
        self,
        max_area: int = 65536,
        m: int = 32,
        interpolation: int = cv2.INTER_AREA,
        p: float = 1.0,
        always_apply: bool | None = None,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.max_area = max_area
        self.m = m
        self.interpolation = interpolation

    def apply(self, img, resize_hw=(0, 0), crop_coords=(0, 0, 0, 0), **params):
        h_r, w_r = resize_hw
        resized = cv2.resize(img, (w_r, h_r), interpolation=self.interpolation)
        x_min, y_min, x_max, y_max = crop_coords
        return resized[y_min:y_max, x_min:x_max]

    def get_params_dependent_on_data(self, params, data):
        h, w = params["shape"][:2]
        (h_r, w_r), (h_c, w_c) = resize_preserve_aspect_area_then_crop(
            h, w, max_area=self.max_area, m=self.m
        )
        shape_key = (h, w)
        if shape_key not in AspectAreaResizeAndCrop._logged_shapes:
            AspectAreaResizeAndCrop._logged_shapes.add(shape_key)
            print(
                f"[AspectAreaResizeAndCrop] ({h}, {w}) → resize ({h_r}, {w_r}) → crop ({h_c}, {w_c})"
            )
        y_min = (h_r - h_c) // 2
        x_min = (w_r - w_c) // 2
        return {
            "resize_hw": (h_r, w_r),
            "crop_coords": (x_min, y_min, x_min + w_c, y_min + h_c),
        }

    def get_transform_init_args_names(self):
        return ("max_area", "m", "interpolation")


# ---------------------------------------------------------------------------
# Step 2 — fractional crop + resize back to pre-crop shape
# ---------------------------------------------------------------------------


class _FractionalCropAndResizeBase(A.DualTransform):
    """Crop a fraction of the input then resize back to the pre-crop shape.

    Concrete subclasses choose the crop origin (random vs center).
    Both the crop fraction and the output shape are parameterised so the
    downstream stages always see a consistent spatial size.
    """

    def __init__(
        self,
        crop_fraction: float = 0.95,
        interpolation: int = cv2.INTER_LINEAR,
        p: float = 1.0,
        always_apply: bool | None = None,
    ):
        super().__init__(p=p, always_apply=always_apply)
        if not 0.0 < crop_fraction <= 1.0:
            raise ValueError("crop_fraction must be in (0.0, 1.0]")
        self.crop_fraction = crop_fraction
        self.interpolation = interpolation

    def apply(
        self,
        img: np.ndarray,
        crop_coords: tuple[int, int, int, int] = (0, 0, 0, 0),
        out_hw: tuple[int, int] = (0, 0),
        **params,
    ) -> np.ndarray:
        x_min, y_min, x_max, y_max = crop_coords
        cropped = img[y_min:y_max, x_min:x_max]
        h_out, w_out = out_hw
        return cv2.resize(cropped, (w_out, h_out), interpolation=self.interpolation)

    def _origin(self, max_y: int, max_x: int) -> tuple[int, int]:
        raise NotImplementedError

    def get_params_dependent_on_data(self, params, data):
        h, w = params["shape"][:2]
        ch = max(1, int(h * self.crop_fraction))
        cw = max(1, int(w * self.crop_fraction))
        y_min, x_min = self._origin(h - ch, w - cw)
        return {
            "crop_coords": (x_min, y_min, x_min + cw, y_min + ch),
            "out_hw": (h, w),
        }

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("crop_fraction", "interpolation")


class FractionalRandomCropAndResize(_FractionalCropAndResizeBase):
    """Random-position fractional crop, then resize back to the pre-crop (H, W)."""

    def _origin(self, max_y: int, max_x: int) -> tuple[int, int]:
        y = np.random.randint(0, max_y + 1) if max_y > 0 else 0
        x = np.random.randint(0, max_x + 1) if max_x > 0 else 0
        return y, x


class FractionalCenterCropAndResize(_FractionalCropAndResizeBase):
    """Center fractional crop, then resize back to the pre-crop (H, W)."""

    def _origin(self, max_y: int, max_x: int) -> tuple[int, int]:
        return max_y // 2, max_x // 2


# ---------------------------------------------------------------------------
# Builder — constructs both train and eval transforms for the processor
# ---------------------------------------------------------------------------


def build_image_transformations_albumentations(
    image_max_area: int = 65536,
    image_resize_m: int = 32,
    random_crop_fraction: float | None = None,
    random_rotation_angle: int | None = None,
    color_jitter_params: dict | None = None,
) -> tuple[A.BaseCompose, A.BaseCompose]:
    """Build the ``(train_transform, eval_transform)`` pair.

    The eval transform is deterministic. The train transform adds stochastic
    augmentation (random-position fractional crop, rotate, color jitter) on
    top of the same deterministic step-1 geometry. When
    ``random_crop_fraction`` is None, step 2 is skipped entirely and the
    pipeline is a pure aspect-area resize + m-aligned center crop.

    Args:
        image_max_area: Area budget for step 1 (default ``256*256 = 65536``).
        image_resize_m: Alignment multiple for step 1 (default ``32``).
        random_crop_fraction: Crop fraction for step 2. None = no-op (no
            step-2 crop, transforms produce step-1 output directly).
        random_rotation_angle: Optional ``A.Rotate(limit=...)`` (train only).
        color_jitter_params: Optional ``A.ColorJitter`` params
            ``{"brightness", "contrast", "saturation", "hue"}`` (train only).

    Returns:
        ``(train_transform, eval_transform)`` — both are albumentations
        composes that take ``image=np.ndarray`` kwargs and return dicts.
    """
    train_list: list = [
        AspectAreaResizeAndCrop(
            max_area=image_max_area, m=image_resize_m, interpolation=cv2.INTER_AREA
        )
    ]
    eval_list: list = [
        AspectAreaResizeAndCrop(
            max_area=image_max_area, m=image_resize_m, interpolation=cv2.INTER_AREA
        )
    ]

    if random_crop_fraction is not None:
        if not 0.0 < random_crop_fraction <= 1.0:
            raise ValueError(
                f"random_crop_fraction must be in (0.0, 1.0], got {random_crop_fraction!r}"
            )
        train_list.append(
            FractionalRandomCropAndResize(
                crop_fraction=random_crop_fraction, interpolation=cv2.INTER_LINEAR
            )
        )
        eval_list.append(
            FractionalCenterCropAndResize(
                crop_fraction=random_crop_fraction, interpolation=cv2.INTER_LINEAR
            )
        )

    if random_rotation_angle is not None and random_rotation_angle != 0:
        train_list.append(A.Rotate(limit=random_rotation_angle, p=1.0))

    if color_jitter_params is not None:
        train_list.append(
            A.ColorJitter(
                brightness=color_jitter_params.get("brightness", 0.0),
                contrast=color_jitter_params.get("contrast", 0.0),
                saturation=color_jitter_params.get("saturation", 0.0),
                hue=color_jitter_params.get("hue", 0.0),
                p=1.0,
            )
        )

    train_transform = A.ReplayCompose(train_list, p=1.0)
    eval_transform = A.Compose(eval_list)
    return train_transform, eval_transform
