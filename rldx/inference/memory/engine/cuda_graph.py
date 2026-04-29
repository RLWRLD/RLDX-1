"""Path B: CUDA Graph capture for GraphSafeMemory.

GraphSafeMemory uses static buffers (position_ids, attention_mask),
so its forward is graph-safe.  We capture it as a single CUDA graph.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CUDAGraphMemoryWrapper(nn.Module):
    """CUDA Graph wrapper for GraphSafeMemory.

    On first call: warms up, captures a CUDA graph.
    On subsequent calls: copies input into static buffer, replays graph,
    returns clone of static output.
    """

    def __init__(self, module, warmup_iters=3):
        super().__init__()
        self.module = module
        self.warmup_iters = warmup_iters
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, inputs_embeds):
        if self.graph is None:
            # Warmup
            for _ in range(self.warmup_iters):
                self.module(inputs_embeds)
            torch.cuda.synchronize()

            # Create static buffers and capture
            self.static_input = inputs_embeds.clone()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_output = self.module(self.static_input)

        self.static_input.copy_(inputs_embeds)
        self.graph.replay()
        return self.static_output.clone()


def setup_cuda_graph(gs_memory):
    """Wrap GraphSafeMemory with CUDA Graph capture.

    Returns:
        CUDAGraphMemoryWrapper ready for benchmark use.
    """
    return CUDAGraphMemoryWrapper(gs_memory)
