# SPDX-License-Identifier: Apache-2.0
"""
StepRequest — typed internal representation of a policy inference request.

Wire protocol stays the same (msgpack `options` dict). This module decodes
that untyped dict into a typed StepRequest once, at the RLDXPolicy boundary,
so downstream code (Validator, Runtime) works on a schema instead of
repeated `options.get("...")` calls.

Options keys consumed (closed set, verified from grep of `rldx_policy.py`):
  - "reset_memory"    — list[bool], per-sample EPISODE reset flags
  - "session_ids"     — list[str], per-sample session identifiers
  - "action_prefix"   — ndarray or Tensor, (B,d,D) or (d,D) for broadcast
  - "rtc_prefix_len"  — int, effective prefix length override

Anything else in options is ignored (legacy or 3rd-party fields).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import numpy as np
    import torch


# ----------------------------------------------------------------------------
# Shape helpers
# ----------------------------------------------------------------------------


def infer_batch_size(obs: dict) -> int:
    """Infer B from batched observation dict.

    RLDXPolicy expects obs with shape {video: {key: (B, T, H, W, C)}, ...}.
    Fall back through video → state → language until we find a batchable
    modality.
    """
    for modality in ("video", "state"):
        d = obs.get(modality)
        if isinstance(d, dict) and d:
            first = next(iter(d.values()))
            shape = getattr(first, "shape", None)
            if shape is not None and len(shape) >= 1:
                return int(shape[0])
    # language: list[list[str]] — batch is outer list
    lang = obs.get("language")
    if isinstance(lang, dict) and lang:
        first = next(iter(lang.values()))
        if isinstance(first, list):
            return len(first)
    raise ValueError(
        "cannot infer batch size from obs: expected video/state/language "
        f"keys, got {sorted(obs.keys()) if isinstance(obs, dict) else type(obs)}"
    )


# ----------------------------------------------------------------------------
# StepRequest
# ----------------------------------------------------------------------------


@dataclass
class StepRequest:
    """Typed policy inference request.

    Created by `decode_options_to_step_request(obs, options)`. Downstream
    consumers never see the raw options dict.
    """

    obs: dict
    batch_size: int
    sids: list[str] | None = None
    reset_mask: list[bool] | None = None
    action_prefix: "np.ndarray | torch.Tensor | None" = None
    rtc_prefix_len: int | None = None

    # Extras that were in options but not consumed — kept for legacy
    # caller debugging, never acted on.
    extras: dict[str, Any] = field(default_factory=dict)

    def validate_shapes(self) -> None:
        """Check per-sample list lengths and action_prefix shape.

        Raises ValueError with actionable message on first violation.
        This replaces scattered runtime checks in _get_action.
        """
        B = self.batch_size

        if self.sids is not None:
            if not isinstance(self.sids, list):
                raise ValueError(f"session_ids must be a list, got {type(self.sids).__name__}")
            if len(self.sids) != B:
                raise ValueError(f"session_ids length {len(self.sids)} != batch size {B}")
            for i, s in enumerate(self.sids):
                if not isinstance(s, str):
                    raise ValueError(f"session_ids[{i}] must be str, got {type(s).__name__}")

        if self.reset_mask is not None:
            if not isinstance(self.reset_mask, list):
                raise ValueError(
                    f"reset_memory must be a list, got {type(self.reset_mask).__name__}"
                )
            if len(self.reset_mask) != B:
                raise ValueError(f"reset_memory length {len(self.reset_mask)} != batch size {B}")
            for i, v in enumerate(self.reset_mask):
                if not isinstance(v, (bool, int)):
                    raise ValueError(f"reset_memory[{i}] must be bool, got {type(v).__name__}")

        if self.action_prefix is not None:
            shape = getattr(self.action_prefix, "shape", None)
            if shape is None:
                raise ValueError(
                    f"action_prefix must have .shape attribute "
                    f"(ndarray or Tensor), got {type(self.action_prefix).__name__}"
                )
            if len(shape) not in (2, 3):
                raise ValueError(
                    f"action_prefix shape must be 2 (d,D) or 3 (B,d,D), got {tuple(shape)}"
                )
            if len(shape) == 3 and shape[0] != B:
                raise ValueError(f"action_prefix batch dim {shape[0]} != batch size {B}")

        if self.rtc_prefix_len is not None:
            if not isinstance(self.rtc_prefix_len, int) or self.rtc_prefix_len < 0:
                raise ValueError(
                    f"rtc_prefix_len must be non-negative int, got {self.rtc_prefix_len!r}"
                )


# ----------------------------------------------------------------------------
# Decoder
# ----------------------------------------------------------------------------

_KNOWN_OPTION_KEYS = frozenset(
    {
        "reset_memory",
        "session_ids",
        "action_prefix",
        "rtc_prefix_len",
    }
)


def decode_options_to_step_request(
    obs: dict,
    options: dict | None,
    *,
    validate: bool = True,
) -> StepRequest:
    """Translate wire-level (obs, options) to typed StepRequest.

    Args:
        obs: Batched observation dict (video/state/language/physics).
        options: Raw options dict from caller; may be None.
        validate: If True, call validate_shapes() before returning.

    Returns:
        StepRequest with populated fields. Unknown option keys go to .extras.
    """
    B = infer_batch_size(obs)

    if options is None:
        req = StepRequest(obs=obs, batch_size=B)
        if validate:
            req.validate_shapes()
        return req

    if not isinstance(options, dict):
        raise ValueError(f"options must be a dict, got {type(options).__name__}")

    reset_mask = options.get("reset_memory")
    if reset_mask is not None and not isinstance(reset_mask, list):
        # accept tuple for convenience, coerce
        reset_mask = list(reset_mask)

    sids = options.get("session_ids")
    if sids is not None and not isinstance(sids, list):
        sids = list(sids)

    extras = {k: v for k, v in options.items() if k not in _KNOWN_OPTION_KEYS}

    req = StepRequest(
        obs=obs,
        batch_size=B,
        sids=sids,
        reset_mask=reset_mask,
        action_prefix=options.get("action_prefix"),
        rtc_prefix_len=options.get("rtc_prefix_len"),
        extras=extras,
    )

    if validate:
        req.validate_shapes()
    return req
