"""Register custom ops for DoubleStreamBlock / ExpandedDoubleStreamBlock.

- ds::fused_attention_2way: RMSNorm + RoPE + Attention [VL|SA]
- ds::fused_attention_3way: RMSNorm + RoPE + Attention [VL|SA|P]
- ds::vl_epilogue_ln: VL residual + LayerNorm fusion
"""

from __future__ import annotations

import double_stream.engine.ops.op_fused_attention  # ds::fused_attention_2way
import double_stream.engine.ops.op_fused_attention_3way  # ds::fused_attention_3way
import double_stream.engine.ops.op_vl_epilogue_ln  # ds::vl_epilogue_ln
