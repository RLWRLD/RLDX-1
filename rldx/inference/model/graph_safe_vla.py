"""Graph-safe unified VLA: backbone + (optional Memory) + Action Head.

Composes GraphSafeQwen3VLBackbone, (optional) GraphSafeMemory, and GraphSafeActionModel
into a single nn.Module for the full VLA pipeline.

Without memory:
  vl_input → VLM → vl_embs(64) → ActionModel → action

With memory (concat_memory=True):
  vl_input → VLM → cog_all(64) → Memory(sliding window + Transformer)
           → [cog_all(64) | cog_augmented(16)] = vl_embs(80)
           → ActionModel → action

Memory cache is managed with pre-allocated static buffers (in-place .copy_())
so that CUDA graph capture and torch.compile are fully compatible.

Public attributes for engine builders:
  gs_backbone:         GraphSafeQwen3VLBackbone   (backbone with vlln)
  gs_action_model: GraphSafeActionModel    (denoising action head)
  gs_memory:      GraphSafeMemory | None (temporal memory module)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GraphSafeVLA(nn.Module):
    """Graph-safe full VLA pipeline with optional memory.

    Forward flow:
      vl_input → VLM → vl_embs
      (optional) vl_embs → Memory → augmented vl_embs
      (vl_embs, state, embodiment_id) → ActionModel → action
    """

    def __init__(self, gs_backbone, gs_action_model, gs_memory=None, memory_config=None):
        super().__init__()
        self.gs_backbone = gs_backbone
        self.gs_action_model = gs_action_model
        self.gs_memory = gs_memory

        # Memory config + static buffers (only when gs_memory is present)
        if gs_memory is not None and memory_config is not None:
            self.n_q = memory_config["n_cog_tokens"]
            self.n_cog_mem = memory_config["memory_n_cog_tokens"]
            self.n_cog_pass = self.n_q - self.n_cog_mem
            self.memory_length = memory_config["memory_length"]
            self.concat_memory = memory_config["concat_memory"]

            # Pre-allocate static cache buffers (B=1, graph-safe)
            K = self.memory_length
            n = self.n_cog_mem
            d = memory_config["hidden_size"]
            device = next(gs_memory.parameters()).device
            dtype = torch.bfloat16

            self.register_buffer(
                "_cached_cog", torch.zeros(1, K * n, d, device=device, dtype=dtype)
            )
            self.register_buffer("_cache_tmp", torch.zeros(1, K * n, d, device=device, dtype=dtype))

    def reset_memory(self):
        """Reset recurrent memory state (call at start of new episode)."""
        if self.gs_memory is not None:
            self._cached_cog.zero_()

    def forward(
        self,
        vl_input,
        state,
        embodiment_id,
        init_noise=None,
        physics_init_noise=None,
        prefix_actions=None,
    ):
        """Full VLA forward.

        Args:
            vl_input: dict with 'pixel_values' (other keys use static buffers)
            state: (B, 1, state_dim)
            embodiment_id: (B,)
            init_noise: (B, action_horizon, action_dim) or None
            physics_init_noise: (B, fut_len, physics_dim) or None
            prefix_actions: (B, prefix_len, action_dim) for RTC trained mode;
                ignored when the action head was built with prefix_len=0.

        Returns:
            action: (B, action_horizon, action_dim)
        """
        vl_embs = self.gs_backbone(vl_input)

        if self.gs_memory is not None:
            vl_embs = self._process_memory(vl_embs)

        return self.gs_action_model(
            vl_embs,
            state,
            embodiment_id,
            init_noise=init_noise,
            physics_init_noise=physics_init_noise,
            prefix_actions=prefix_actions,
        )

    def _process_memory(self, vl_embs):
        """Process VLM output through memory (sliding window + TransformerMemory).

        Uses in-place .copy_() on pre-allocated static buffers for CUDA graph
        and torch.compile compatibility. No tensor allocation in the forward path.

        Args:
            vl_embs: (B, n_q, d) — backbone cog tokens

        Returns:
            concat_memory=True:  (B, n_q + n_cog_mem, d)
            concat_memory=False: (B, n_q, d)
        """
        cog_all = vl_embs[:, -self.n_q :, :]  # (B, n_q, d)
        cog_current = cog_all[:, self.n_cog_pass :, :]  # (B, n_cog_mem, d)

        n = self.n_cog_mem

        # Shift left into tmp buffer, append current, copy back
        # All in-place — _cached_cog data_ptr never changes
        self._cache_tmp[:, :-n, :].copy_(self._cached_cog[:, n:, :])
        self._cache_tmp[:, -n:, :].copy_(cog_current)
        self._cached_cog.copy_(self._cache_tmp)

        # TransformerMemory forward
        memory_out = self.gs_memory(self._cached_cog)  # (B, K*n_cog_mem, d)
        cog_augmented = memory_out[:, -n:, :]  # (B, n_cog_mem, d)

        # Assemble output
        if self.concat_memory:
            return torch.cat([cog_all, cog_augmented], dim=1)
        else:
            if self.n_cog_pass > 0:
                return torch.cat([cog_all[:, : self.n_cog_pass, :], cog_augmented], dim=1)
            return cog_augmented
