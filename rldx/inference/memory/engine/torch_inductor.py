"""torch.compile optimization for Memory module.

Compiles GraphSafeMemory with torch.compile.
For custom Triton kernel chains, see custom_memory_chain.py.
"""

from __future__ import annotations

import time as _time

import torch


def setup_compile(gs_memory, sample_input, mode="max-autotune"):
    """Compile memory module with torch.compile.

    Args:
        gs_memory: GraphSafeMemory instance
        sample_input: sample inputs_embeds tensor (B, S, D)
        mode: torch.compile mode

    Returns:
        (compiled_memory, compile_time_s)
    """
    compiled_memory = torch.compile(gs_memory, mode=mode)

    # Trigger compilation
    t0 = _time.time()
    with torch.no_grad():
        compiled_memory(sample_input)
    torch.cuda.synchronize()
    compile_time = _time.time() - t0

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            compiled_memory(sample_input)
    torch.cuda.synchronize()

    return compiled_memory, compile_time
