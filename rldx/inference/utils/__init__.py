"""Shared utilities for the inference benchmarks.

Re-exports through PEP 562 ``__getattr__`` so tests that only need the
torch-free helpers (``MODEL_REGISTRY``, ``is_placeholder_path``) don't
pull torch / transformers transitively.
"""

from __future__ import annotations


_LAZY = {
    "MODEL_REGISTRY": ("registry", "MODEL_REGISTRY"),
    "is_placeholder_path": ("registry", "is_placeholder_path"),
    "measure_times": ("timing", "measure_times"),
    "print_latency_table": ("timing", "print_latency_table"),
    "cos_sim": ("correctness", "cos_sim"),
    "print_correctness": ("correctness", "print_correctness"),
    "print_diff": ("correctness", "print_diff"),
    "load_backbone": ("loader", "load_backbone"),
    "load_action_model": ("loader", "load_action_model"),
    "load_memory": ("loader", "load_memory"),
    "load_vla": ("loader", "load_vla"),
    "generate_synthetic_input": ("input_generator", "generate_synthetic_input"),
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module

        submod, attr = _LAZY[name]
        value = getattr(import_module(f"{__name__}.{submod}"), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_LAZY))
