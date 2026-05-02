"""Register all custom ops for TransformerMemory.

Import this module to register all mem:: ops at once.
"""

from __future__ import annotations

import memory.engine.ops.op_fused_epilogue_add2_rmsnorm  # mem::fused_epilogue_add2_rmsnorm
import memory.engine.ops.op_fused_memory_attention  # mem::fused_attention
