"""Inference acceleration suite for RLDX models.

The public entry point is ``rldx.inference.serve_optimization.apply_optimization``,
which installs an optimization path (A/B/C/D) onto a loaded ``RLDXPolicy``.
"""

from __future__ import annotations

import os as _os
import sys as _sys


# Cross-module imports inside this package use bare paths
# (``from action_model.X import …``, ``from backbone.engine.Y import …``,
# ``from model import Z``) so they resolve both as a regular
# ``rldx.inference.*`` import and when a benchmark script is launched
# directly. Prime sys.path once on first import to support both.
_INF_ROOT = _os.path.dirname(_os.path.abspath(__file__))
for _sub in (
    "action_model",
    "action_model/engine",
    "action_model/single_stream",
    "action_model/single_stream/engine",
    "action_model/double_stream",
    "action_model/double_stream/engine",
    "backbone",
    "backbone/engine",
    "backbone/llm/engine",
    "backbone/vision_encoder/engine",
    "memory",
    "memory/engine",
    "engine",
    "",
):
    _d = _os.path.join(_INF_ROOT, _sub) if _sub else _INF_ROOT
    if _os.path.isdir(_d) and _d not in _sys.path:
        _sys.path.insert(0, _d)

del _os, _sys, _INF_ROOT, _sub, _d
