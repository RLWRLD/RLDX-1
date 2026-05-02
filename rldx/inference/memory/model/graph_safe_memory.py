"""Graph-safe wrapper for TransformerMemory.

Pre-computes data-dependent values (position_ids, attention_mask) in __init__,
then uses them as static buffers in forward.

The cache management (shift/append of past cog tokens across timesteps) is
stateful across calls and must be handled externally — this module only wraps
the fixed-shape Transformer forward.

Architecture:
  TransformerMemory receives (B, K * n_cog_mem, d) flattened cog tokens from
  K timesteps and returns memory-augmented hidden states of the same shape.
  The last n_cog_mem tokens (current timestep) are the augmented output.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rldx.utils.dist import rank_zero_print as _print


class GraphSafeMemory(nn.Module):
    """Graph-safe wrapper for TransformerMemory with static buffers.

    Pre-computes at __init__ time:
      - position_ids: block-wise positions (indices // block_attn_size)
      - attention_mask: causal or block attention mask

    Args:
        memory_module: TransformerMemory instance
        memory_length: number of timesteps (K)
        memory_n_cog_tokens: number of cog tokens per timestep routed through memory
        device: target device
        dtype: data type for attention mask
    """

    def __init__(
        self, memory_module, memory_length, memory_n_cog_tokens, device=None, dtype=torch.bfloat16
    ):
        super().__init__()
        self._memory = memory_module
        self.memory_length = memory_length
        self.memory_n_cog_tokens = memory_n_cog_tokens

        seq_length = memory_length * memory_n_cog_tokens
        block_attn_size = memory_module.block_attn_size
        use_causal_attn = memory_module.use_causal_attn

        # --- Static position IDs (block-wise: position = index // block_size) ---
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids // block_attn_size
        # (1, seq_length) for broadcast over batch
        self.register_buffer("static_position_ids", position_ids.unsqueeze(0))

        # --- Static attention mask ---
        if seq_length > 1:
            attn_mask = self._make_mask(seq_length, block_attn_size, use_causal_attn, device, dtype)
            self.register_buffer("static_attention_mask", attn_mask)
        else:
            self.static_attention_mask = None

        _print(
            f"  [GraphSafeMemory] seq_length={seq_length}, "
            f"block_attn_size={block_attn_size}, "
            f"K={memory_length}, n_cog_mem={memory_n_cog_tokens}, "
            f"causal={use_causal_attn}"
        )

    @staticmethod
    def _make_mask(seq_length, block_attn_size, use_causal_attn, device, dtype):
        """Pre-compute attention mask (same logic as memory._make_causal_mask)."""
        if use_causal_attn:
            mask = torch.full(
                (seq_length, seq_length), torch.finfo(dtype).min, device=device, dtype=dtype
            )
            cond = torch.arange(seq_length, device=device)
            mask.masked_fill_(cond < (cond + 1).view(seq_length, 1), 0)
        else:
            # Block attention: each block attends to itself and all previous blocks
            mask = torch.full(
                (seq_length, seq_length), torch.finfo(dtype).min, device=device, dtype=dtype
            )
            for i in range(0, seq_length, block_attn_size):
                end_i = min(i + block_attn_size, seq_length)
                mask[i:end_i, :end_i] = 0

        # Expand to (1, 1, seq, seq) for attention broadcasting
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, inputs_embeds):
        """Graph-safe memory forward.

        Args:
            inputs_embeds: (B, K * n_cog_mem, d) — flattened cog tokens from K timesteps

        Returns:
            (B, K * n_cog_mem, d) — memory-augmented hidden states.
            Caller should extract the last n_cog_mem tokens for the augmented output.
        """
        B = inputs_embeds.shape[0]
        memory = self._memory

        hidden_states = inputs_embeds

        # Add sinusoidal positional embeddings if not using RoPE
        if not memory.use_rope and hasattr(memory, "pos_emb"):
            hidden_states = memory.pos_emb(hidden_states)

        # Expand static position_ids to batch
        position_ids = self.static_position_ids.expand(B, -1)

        # Process through transformer layers
        for decoder_layer in memory.layers:
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=self.static_attention_mask,
                position_ids=position_ids,
                use_rope=memory.use_rope,
            )

        hidden_states = memory.norm(hidden_states)
        return hidden_states
