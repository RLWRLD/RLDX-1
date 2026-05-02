"""Path bootstrap for benchmark scripts launched directly (not via ``pip install -e .``).

Usage:
    import _path; _path.setup(__file__)
"""

from __future__ import annotations

import os
import sys


INFERENCE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(os.path.dirname(INFERENCE_DIR))


def setup(caller_file=None):
    """Add the repo root + inference dir to sys.path; pin caller's dir at index 0."""
    for p in (PROJ_ROOT, INFERENCE_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    if caller_file is not None:
        caller_dir = os.path.dirname(os.path.abspath(caller_file))
        if caller_dir in sys.path:
            sys.path.remove(caller_dir)
        sys.path.insert(0, caller_dir)
