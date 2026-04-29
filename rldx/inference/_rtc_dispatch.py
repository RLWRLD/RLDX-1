"""Pure-Python RTC bake-decision helpers.

Lives in its own module so the dispatcher policy can be unit-tested
without dragging torch / triton / cuda into the test environment.
"""

from __future__ import annotations


def resolve_rtc_for_bake(full_model, path: str) -> int:
    """Return the trained-mode prefix length to bake into the chain.

    Returns 0 unless ``path`` is C or D and the model is configured for
    trained-mode RTC with a positive ``rtc_inference_delay``.
    """
    if path not in ("C", "D"):
        return 0
    cfg = getattr(full_model, "config", None)
    if cfg is None:
        return 0
    if getattr(cfg, "rtc_inference_mode", "none") != "trained":
        return 0
    return max(int(getattr(cfg, "rtc_inference_delay", 0) or 0), 0)
