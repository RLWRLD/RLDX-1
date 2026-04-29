"""torch.compile optimization for VLA pipeline (component-wise).

Compiles VLM, Memory, and ActionModel sub-modules independently,
then lets GraphSafeVLA forward connect them in eager mode.

Strategy:
  - VLM:        torch.compile(gs_backbone.forward)
  - Memory:     torch.compile(gs_memory) (if present)
  - ActionModel: torch.compile each sub-module
                (gs_msat, action_encoder, state_encoder, vlln, action_decoder)
  - Glue code:  eager (memory processing, denoising loop)

For the unified custom Triton chain, see custom_vla_chain.py.
"""

from __future__ import annotations

import time as _time

import torch

from rldx.utils.dist import rank_zero_print as _print


def setup_vla_compile(
    gs_vla,
    vl_input,
    state,
    embodiment_id,
    init_noise=None,
    physics_init_noise=None,
    mode="max-autotune",
):
    """Compile all sub-modules of the VLA pipeline.

    Args:
        gs_vla: GraphSafeVLA instance
        vl_input: sample VLM input dict
        state: sample state tensor (B, N_state, state_dim)
        embodiment_id: sample embodiment ID tensor (B,)
        init_noise: sample action noise
        physics_init_noise: sample physics noise (optional, v1.4-all)
        mode: torch.compile mode

    Returns:
        (orig_modules, total_compile_time_s)
    """
    vlm = gs_vla.gs_backbone
    action_head = gs_vla.gs_action_model

    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.reset()

    total_t0 = _time.time()
    orig_modules = {}

    # --- Compile VLM ---
    _print("  Compiling VLM...")
    orig_modules["vlm_forward"] = vlm.forward
    compiled_vlm = torch.compile(vlm.forward, mode=mode)
    vlm.forward = compiled_vlm

    t0 = _time.time()
    with torch.no_grad():
        vlm(vl_input)
    torch.cuda.synchronize()
    vlm_time = _time.time() - t0
    _print(f"    VLM compilation: {vlm_time:.1f}s")

    for _ in range(3):
        with torch.no_grad():
            vlm(vl_input)
    torch.cuda.synchronize()

    # --- Compile Memory (if present) ---
    if gs_vla.gs_memory is not None:
        _print("  Compiling Memory...")
        orig_modules["gs_memory"] = gs_vla.gs_memory
        compiled_memory = torch.compile(gs_vla.gs_memory, mode=mode)
        gs_vla.gs_memory = compiled_memory

        # Trigger compilation
        with torch.no_grad():
            vl_embs = vlm(vl_input).clone()
            gs_vla._process_memory(vl_embs)
        torch.cuda.synchronize()
        _print("    Memory compiled")

    # --- Compile ActionModel sub-modules ---
    _print("  Compiling ActionModel sub-modules...")
    orig_modules["gs_msat"] = action_head.gs_msat
    orig_modules["action_encoder"] = action_head.action_encoder
    orig_modules["state_encoder"] = action_head.state_encoder
    orig_modules["vlln"] = action_head.vlln
    orig_modules["action_decoder"] = action_head.action_decoder

    action_head.gs_msat = torch.compile(action_head.gs_msat, mode=mode)
    action_head.action_encoder = torch.compile(action_head.action_encoder, mode=mode)
    action_head.state_encoder = torch.compile(action_head.state_encoder, mode=mode)
    action_head.vlln = torch.compile(action_head.vlln, mode=mode)
    action_head.action_decoder = torch.compile(action_head.action_decoder, mode=mode)

    # Trigger ActionModel compilation via full forward
    with torch.no_grad():
        vl_embs = vlm(vl_input).clone()
        if gs_vla.gs_memory is not None:
            vl_embs = gs_vla._process_memory(vl_embs)

    t0 = _time.time()
    with torch.no_grad():
        action_head(
            vl_embs,
            state,
            embodiment_id,
            init_noise=init_noise,
            physics_init_noise=physics_init_noise,
        )
    torch.cuda.synchronize()
    ah_time = _time.time() - t0
    _print(f"    ActionModel compilation: {ah_time:.1f}s")

    for _ in range(3):
        with torch.no_grad():
            action_head(
                vl_embs,
                state,
                embodiment_id,
                init_noise=init_noise,
                physics_init_noise=physics_init_noise,
            )
    torch.cuda.synchronize()

    total_time = _time.time() - total_t0
    _print(f"  Total compilation: {total_time:.1f}s")

    return orig_modules, total_time


def restore_vla_compile(gs_vla, orig_modules):
    """Restore original modules after compiled benchmark."""
    if "vlm_forward" in orig_modules:
        gs_vla.gs_backbone.forward = orig_modules["vlm_forward"]
    if "gs_memory" in orig_modules:
        gs_vla.gs_memory = orig_modules["gs_memory"]

    action_head = gs_vla.gs_action_model
    for key in ("gs_msat", "action_encoder", "state_encoder", "vlln", "action_decoder"):
        if key in orig_modules:
            setattr(action_head, key, orig_modules[key])

    torch._dynamo.reset()
