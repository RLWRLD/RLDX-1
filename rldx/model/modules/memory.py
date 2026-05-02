"""
Memory module for cognition tokens with temporal context aggregation.

This module implements a Transformer-based memory that fuses cognition token embeddings
from multiple timesteps to provide temporal context for action prediction.
"""

from typing import Optional

from diffusers.models.embeddings import SinusoidalPositionalEmbedding
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast

from rldx.model.modules.norms import RMSNorm
from rldx.utils.dist import rank_zero_print as _print


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        # x: [batch_size, num_heads, seq_len, head_dim]
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary position embedding to query and key tensors."""
    cos = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    sin = sin.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional Grouped Query Attention (GQA) support."""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_rope: bool = True,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if use_rope:
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Repeat k/v heads for GQA
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output


class SwiGLUMLP(nn.Module):
    """MLP module with SwiGLU activation."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with pre-normalization."""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MultiHeadAttention(config, layer_idx)
        self.mlp = SwiGLUMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_rope: bool = True,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_rope=use_rope,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


def _make_causal_mask(
    input_shape: tuple,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
    block_attn_size: int = 1,
    use_causal_attn: bool = True,
) -> torch.Tensor:
    """
    Create causal attention mask.

    Args:
        input_shape: (batch_size, seq_length)
        dtype: Data type for the mask
        device: Device to create mask on
        past_key_values_length: Length of past key values
        block_attn_size: Size of blocks for block-wise attention
        use_causal_attn: If True, use causal mask; if False, use block attention
    """
    bsz, tgt_len = input_shape

    if use_causal_attn:
        # Standard causal mask
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(tgt_len, device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(tgt_len, 1), 0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat(
                [torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask],
                dim=-1,
            )
    else:
        # Block attention: each block can only attend to itself and previous blocks
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        for i in range(0, tgt_len, block_attn_size):
            end_i = min(i + block_attn_size, tgt_len)
            # Allow attention within block and to all previous blocks
            mask[i:end_i, :end_i] = 0
        mask = mask.to(dtype)

    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


class TransformerMemory(nn.Module):
    """
    Transformer-based memory module for temporal aggregation of cognition token embeddings.

    This module processes a sequence of cognition token embeddings from multiple timesteps
    and produces a memory-augmented representation using causal self-attention.
    """

    def __init__(
        self,
        hidden_size: int = 1536,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        max_position_embeddings: int = 8,
        rms_norm_eps: float = 1e-5,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        use_causal_attn: bool = True,
        use_rope: bool = True,
        block_attn_size: int = 1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.use_causal_attn = use_causal_attn
        self.use_rope = use_rope
        self.block_attn_size = block_attn_size

        # Create LlamaConfig for internal use
        self.config = LlamaConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
        )

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(self.config, layer_idx)
                for layer_idx in range(num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Optional sinusoidal positional embedding (if not using RoPE)
        if not use_rope:
            self.pos_emb = SinusoidalPositionalEmbedding(
                hidden_size, max_seq_length=max_position_embeddings
            )

        # Initialize weights
        self.apply(self._init_weights)

        _print(
            f"\n[i] Transformer-based memory module initialized: "
            f"(layers={num_hidden_layers}, heads={num_attention_heads}, "
            f"hidden_size={hidden_size}, causal={use_causal_attn}, rope={use_rope})"
        )

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> BaseModelOutputWithPast:
        """
        Forward pass through the transformer memory.

        Args:
            inputs_embeds: Input embeddings of shape (batch_size, seq_length, hidden_size)
            attention_mask: Optional attention mask
            position_ids: Optional position IDs

        Returns:
            BaseModelOutputWithPast with last_hidden_state of shape (batch_size, seq_length, hidden_size)
        """
        batch_size, seq_length, _ = inputs_embeds.shape
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            # Divide by block_attn_size for block-wise position encoding
            position_ids = (
                (position_ids // self.block_attn_size).unsqueeze(0).expand(batch_size, -1)
            )

        # Create causal/block attention mask
        if seq_length > 1:
            causal_mask = _make_causal_mask(
                (batch_size, seq_length),
                dtype=dtype,
                device=device,
                block_attn_size=self.block_attn_size,
                use_causal_attn=self.use_causal_attn,
            )
        else:
            causal_mask = None

        hidden_states = inputs_embeds

        # Add sinusoidal positional embeddings if not using RoPE
        if not self.use_rope and hasattr(self, "pos_emb"):
            hidden_states = self.pos_emb(hidden_states)

        # Process through transformer layers
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                use_rope=self.use_rope,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
