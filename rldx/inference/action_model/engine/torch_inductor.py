"""torch.compile optimization for Action Head pipeline.

Two compilation strategies:
  1. setup_msat_compile: compile GraphSafeMSAT as a single unit
  2. setup_action_model_compile: compile all sub-modules independently
     (gs_msat, action_encoder, state_encoder, vlln, action_decoder)

For custom Triton kernel chains, see custom_action_model_chain.py.
"""

from __future__ import annotations

import time as _time

import torch

from rldx.utils.dist import rank_zero_print as _print


def setup_msat_compile(gs_msat, mode="max-autotune"):
    """Compile GraphSafeMSAT with torch.compile.

    Args:
        gs_msat: GraphSafeMSAT instance
        mode: torch.compile mode

    Returns:
        compiled_msat -- drop-in replacement for gs_action_model.gs_msat
    """
    return torch.compile(gs_msat, mode=mode)


def setup_action_model_compile(
    gs_action_model,
    vl_embs,
    state,
    embodiment_id,
    init_noise=None,
    physics_hist=None,
    physics_init_noise=None,
    mode="max-autotune",
):
    """Compile all sub-modules of GraphSafeActionModel.

    Compiles: gs_msat, action_encoder, state_encoder, vlln, action_decoder.
    The denoising loop and control flow stay eager.

    Args:
        gs_action_model: GraphSafeActionModel instance
        vl_embs: sample VL embeddings (B, N_vl, D)
        state: sample state (B, N_state, state_dim)
        embodiment_id: sample embodiment ID (B,)
        init_noise: sample noise (B, action_horizon, action_dim)
        physics_hist: sample physics history (optional, v1.4-all)
        physics_init_noise: sample physics noise (optional, v1.4-all)
        mode: torch.compile mode

    Returns:
        (orig_modules, compile_time_s)
    """
    orig_modules = {
        "gs_msat": gs_action_model.gs_msat,
        "action_encoder": gs_action_model.action_encoder,
        "state_encoder": gs_action_model.state_encoder,
        "vlln": gs_action_model.vlln,
        "action_decoder": gs_action_model.action_decoder,
    }

    torch._dynamo.reset()
    t0 = _time.time()

    _print(f"  Compiling action head sub-modules (mode={mode})...")
    gs_action_model.gs_msat = torch.compile(gs_action_model.gs_msat, mode=mode)
    gs_action_model.action_encoder = torch.compile(gs_action_model.action_encoder, mode=mode)
    gs_action_model.state_encoder = torch.compile(gs_action_model.state_encoder, mode=mode)
    gs_action_model.vlln = torch.compile(gs_action_model.vlln, mode=mode)
    gs_action_model.action_decoder = torch.compile(gs_action_model.action_decoder, mode=mode)

    # Trigger compilation
    with torch.no_grad():
        gs_action_model(
            vl_embs,
            state,
            embodiment_id,
            init_noise=init_noise,
            physics_hist=physics_hist,
            physics_init_noise=physics_init_noise,
        )
    torch.cuda.synchronize()
    compile_time_s = _time.time() - t0
    _print(f"  Action head compilation: {compile_time_s:.1f}s")

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            gs_action_model(
                vl_embs,
                state,
                embodiment_id,
                init_noise=init_noise,
                physics_hist=physics_hist,
                physics_init_noise=physics_init_noise,
            )
    torch.cuda.synchronize()

    return orig_modules, compile_time_s


def restore_action_model_compile(gs_action_model, orig_modules):
    """Restore original sub-modules after compiled benchmark."""
    for key, module in orig_modules.items():
        setattr(gs_action_model, key, module)
    torch._dynamo.reset()
