"""Path B: CUDA Graph capture for GraphSafeMSAT.

With GraphSafeMSAT applied, the MSAT forward is graph-safe
(static RoPE IDs, no conditional branching).  We wrap it with
CUDAGraphModuleWrapper for the denoising loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CUDAGraphModuleWrapper(nn.Module):
    """Generic CUDA Graph wrapper for nn.Module with fixed input shapes.

    On first call: warms up, then captures a CUDA graph.
    On subsequent calls: copies inputs into static buffers, replays the graph,
    and returns clone(s) of the static output.
    """

    def __init__(self, module, warmup_iters=3):
        super().__init__()
        self.module = module
        self.warmup_iters = warmup_iters
        self.graph = None
        self.static_args = None
        self.static_kwargs = None
        self.static_output = None

    def forward(self, *args, **kwargs):
        if self.graph is None:
            for _ in range(self.warmup_iters):
                self.module(*args, **kwargs)
            torch.cuda.synchronize()

            self.static_args = [a.clone() if isinstance(a, torch.Tensor) else a for a in args]
            self.static_kwargs = {
                k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()
            }

            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_output = self.module(*self.static_args, **self.static_kwargs)

        for i, a in enumerate(args):
            if isinstance(a, torch.Tensor):
                self.static_args[i].copy_(a)
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                self.static_kwargs[k].copy_(v)

        self.graph.replay()
        if isinstance(self.static_output, dict):
            return {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in self.static_output.items()
            }
        if isinstance(self.static_output, tuple):
            return tuple(o.clone() for o in self.static_output)
        return self.static_output.clone()


def setup_cuda_graph(gs_msat):
    """Wrap GraphSafeMSAT with CUDA Graph capture.

    Returns:
        cuda_graph_msat — CUDAGraphModuleWrapper ready for pipeline use
    """
    return CUDAGraphModuleWrapper(gs_msat)
