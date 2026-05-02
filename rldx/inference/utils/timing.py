"""Latency reporting and timing utilities for E2E benchmarks."""

from __future__ import annotations

import torch

from rldx.utils.dist import rank_zero_print as _print


def measure_times(fn, iters):
    """Measure kernel execution times with CUDA events."""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    torch.cuda.synchronize()
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [float(s.elapsed_time(e)) for s, e in zip(starts, ends)]


def print_latency_table(title, results_dict):
    """Print a latency results table with p50/mean/std/speedup."""
    _print(f"\n{title}:")
    hdr = f"{'Path':<45} {'p50 (ms)':>10} {'mean (ms)':>10} {'std (ms)':>10} {'speedup':>8}"
    _print(hdr)
    _print("-" * len(hdr))
    base_p50 = None
    for label, times in results_dict.items():
        t = torch.tensor(times, dtype=torch.float64)
        p50, mean, std = t.quantile(0.5).item(), t.mean().item(), t.std().item()
        if base_p50 is None:
            base_p50 = p50
        sp = f"{base_p50 / p50:.2f}x" if base_p50 != p50 else ""
        _print(f"{label:<45} {p50:>10.4f} {mean:>10.4f} {std:>10.4f} {sp:>8}")
