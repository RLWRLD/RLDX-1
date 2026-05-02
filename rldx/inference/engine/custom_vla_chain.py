"""Unified VLA chain: single nn.Module composing VLM + (Memory) + ActionModel.

Builds both chains internally from GraphSafeVLA, assembles into one Module.
Designed to be compiled as a single torch.compile unit for cross-component optimization.

Selects 2-way or 3-way ActionModel chain based on n_physics:
  v1.4-v:   CustomActionHeadChain (no physics)
  v1.4-all: CustomExpandedActionHeadChain (with physics denoising)

Memory module is included when GraphSafeVLA has gs_memory.
"""

from __future__ import annotations

from action_model.engine.custom_action_model_chain import build_custom_action_model_chain
from action_model.engine.custom_expanded_action_model_chain import (
    build_custom_expanded_action_model_chain,
)
from backbone.engine.custom_backbone_chain import build_custom_backbone_chain
import torch
import torch.nn as nn

from rldx.utils.dist import rank_zero_print as _print


class CustomVLAChain(nn.Module):
    """Unified VLA pipeline: VLM + (Memory) + ActionModel.

    When gs_memory is provided, processes VLM output through memory
    before passing to ActionModel (matching GraphSafeVLA._process_memory).
    """

    def __init__(self, backbone_chain, action_model_chain, gs_vla=None):
        super().__init__()
        self.backbone_chain = backbone_chain
        self.action_model_chain = action_model_chain

        # Memory (optional, from GraphSafeVLA)
        self.has_memory = gs_vla is not None and gs_vla.gs_memory is not None
        if self.has_memory:
            self.gs_memory = gs_vla.gs_memory
            self.n_q = gs_vla.n_q
            self.n_cog_mem = gs_vla.n_cog_mem
            self.n_cog_pass = gs_vla.n_cog_pass
            self.memory_length = gs_vla.memory_length
            self.concat_memory = gs_vla.concat_memory
            # Share cache state with GraphSafeVLA
            self._gs_vla = gs_vla

    def _process_memory(self, vl_embs):
        """Process VLM output through memory (delegates to GraphSafeVLA)."""
        return self._gs_vla._process_memory(vl_embs)

    def forward(self, pixel_values, state, embodiment_id, init_noise=None, prefix_actions=None):
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])
        vl_embs = self.backbone_chain(pixel_values)

        if self.has_memory:
            vl_embs = self._process_memory(vl_embs)

        return self.action_model_chain(
            vl_embs,
            state,
            embodiment_id,
            init_noise=init_noise,
            prefix_actions=prefix_actions,
        )


class CustomExpandedVLAChain(nn.Module):
    """Unified VLA pipeline with physics: VLM + (Memory) + ExpandedActionHead."""

    def __init__(self, backbone_chain, action_model_chain, gs_vla=None):
        super().__init__()
        self.backbone_chain = backbone_chain
        self.action_model_chain = action_model_chain

        self.has_memory = gs_vla is not None and gs_vla.gs_memory is not None
        if self.has_memory:
            self._gs_vla = gs_vla

    def _process_memory(self, vl_embs):
        return self._gs_vla._process_memory(vl_embs)

    def forward(
        self,
        pixel_values,
        state,
        embodiment_id,
        init_noise=None,
        physics_hist=None,
        physics_init_noise=None,
        prefix_actions=None,
    ):
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])
        vl_embs = self.backbone_chain(pixel_values)

        if self.has_memory:
            vl_embs = self._process_memory(vl_embs)

        return self.action_model_chain(
            vl_embs,
            state,
            embodiment_id,
            init_noise=init_noise,
            physics_hist=physics_hist,
            physics_init_noise=physics_init_noise,
            prefix_actions=prefix_actions,
        )


def build_custom_vla_chain(gs_vla, device, dtype=torch.bfloat16, bake_prefix_len=0):
    """Build a CustomVLAChain from a GraphSafeVLA (no compilation).

    Auto-selects 2-way or 3-way ActionModel chain. Includes memory if present.
    The trained-mode prefix length is read off ``gs_vla.gs_action_model``;
    ``bake_prefix_len`` is accepted as an explicit assertion that the
    caller's expected prefix matches the substrate.
    """
    expected = getattr(gs_vla.gs_action_model, "prefix_len", 0)
    if bake_prefix_len and bake_prefix_len != expected:
        raise ValueError(
            f"build_custom_vla_chain: bake_prefix_len={bake_prefix_len} "
            f"but gs_action_model.prefix_len={expected}; the substrate must "
            "be constructed with the same prefix length."
        )

    backbone_chain = build_custom_backbone_chain(gs_vla.gs_backbone, device=device, dtype=dtype)

    use_physics = getattr(gs_vla.gs_action_model, "use_physics", False)
    n_physics = getattr(gs_vla.gs_action_model.gs_msat, "n_physics", 0)

    if use_physics and n_physics > 0:
        ah_chain = build_custom_expanded_action_model_chain(
            gs_vla.gs_action_model, device=device, dtype=dtype
        )
        return CustomExpandedVLAChain(backbone_chain, ah_chain, gs_vla=gs_vla).eval()
    ah_chain = build_custom_action_model_chain(gs_vla.gs_action_model, device=device, dtype=dtype)
    return CustomVLAChain(backbone_chain, ah_chain, gs_vla=gs_vla).eval()


def compile_custom_vla_chain(vla_chain, sample_inputs, compile_mode="max-autotune", fullgraph=True):
    """Compile a CustomVLAChain with torch.compile and trigger compilation.

    ``sample_inputs`` is the positional + keyword tuple the chain forward
    expects; an optional trailing ``prefix_actions`` is recognised and
    forwarded when the substrate was built with ``prefix_len > 0``.

    ``fullgraph=True`` is the default — the chain is engineered to be
    graph-break-free so that Inductor can fuse the entire VLA forward
    into a single FX graph (and, with ``mode='max-autotune'``, wrap it
    in a single CUDA Graph at replay time).
    """
    import time as _time

    _print(f"  [VLAChain] Compiling ({compile_mode}, fullgraph={fullgraph})...")
    compiled_chain = torch.compile(vla_chain, mode=compile_mode, fullgraph=fullgraph)

    prefix_actions = None
    if isinstance(vla_chain, CustomExpandedVLAChain):
        if len(sample_inputs) == 7:
            (
                pixel_values,
                state,
                embodiment_id,
                init_noise,
                physics_hist,
                physics_init_noise,
                prefix_actions,
            ) = sample_inputs
        else:
            (pixel_values, state, embodiment_id, init_noise, physics_hist, physics_init_noise) = (
                sample_inputs
            )
        t0 = _time.time()
        with torch.no_grad():
            compiled_chain(
                pixel_values,
                state,
                embodiment_id,
                init_noise=init_noise,
                physics_hist=physics_hist,
                physics_init_noise=physics_init_noise,
                prefix_actions=prefix_actions,
            )
    else:
        if len(sample_inputs) == 5:
            pixel_values, state, embodiment_id, init_noise, prefix_actions = sample_inputs
        else:
            pixel_values, state, embodiment_id, init_noise = sample_inputs
        t0 = _time.time()
        with torch.no_grad():
            compiled_chain(
                pixel_values,
                state,
                embodiment_id,
                init_noise=init_noise,
                prefix_actions=prefix_actions,
            )

    torch.cuda.synchronize()
    compile_time_s = _time.time() - t0
    _print(f"  [VLAChain] Compilation: {compile_time_s:.1f}s")

    return compiled_chain, compile_time_s
