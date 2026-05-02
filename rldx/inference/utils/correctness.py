"""Correctness reporting for E2E benchmarks."""

from __future__ import annotations

import torch

from rldx.utils.dist import rank_zero_print as _print


def cos_sim(a, b):
    """Cosine similarity between two tensors (flattened to 1-D)."""
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    return torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def print_diff(label, ref, test, indent="  "):
    """Compare two feature tensors. Handles shape mismatch."""
    if ref.shape != test.shape:
        _print(
            f"{indent}{label}: SHAPE MISMATCH eager={list(ref.shape)} vs path={list(test.shape)}"
        )
        return
    d = (ref - test).abs()
    rel = d.max().item() / (ref.abs().max().item() + 1e-8)
    _print(
        f"{indent}{label}: max_diff={d.max().item():.6f}, rel={rel:.6f}, "
        f"allclose(atol=1e-2)={torch.allclose(ref, test, atol=1e-2)}"
    )


def print_correctness(title, entries, eager_output):
    """Print correctness comparison table.

    entries: list of (label, output_tensor) or (label, output_tensor, error_str).
    eager_output: reference tensor from eager baseline.
    """
    _print(f"\n{title}:")
    for entry in entries:
        if len(entry) == 3:
            label, output, err = entry
        else:
            label, output = entry
            err = None
        if err is not None:
            _print(f"  {label:<22} {err}")
        elif output is None:
            _print(f"  {label:<22} NOT AVAILABLE")
        elif eager_output.shape != output.shape:
            _print(
                f"  {label:<22} SHAPE MISMATCH eager={list(eager_output.shape)} "
                f"vs path={list(output.shape)}"
            )
        else:
            d = (eager_output - output).abs()
            cs = cos_sim(eager_output, output)
            ac = torch.allclose(eager_output, output, atol=1e-2)
            _print(f"  {label:<22} max_diff={d.max().item():.6f}  cos_sim={cs:.8f}  allclose={ac}")
