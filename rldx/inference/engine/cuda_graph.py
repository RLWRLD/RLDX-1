"""Path D: CUDA Graph capture for full VLA pipeline.

Captures GraphSafeVLA.forward() (VLM + ActionModel denoising loop) as a
single CUDA graph.  All data-dependent operations are pre-resolved by the
GraphSafe wrappers, so the entire pipeline is graph-safe.

Note: torch.randn in the ActionModel denoising loop is captured with fixed RNG
state — replayed noise will differ between captures but is deterministic
within a captured graph.
"""

from __future__ import annotations

import torch

from rldx.utils.dist import rank_zero_print as _print


def setup_vla_cuda_graph(
    gs_vla,
    vl_input,
    state,
    embodiment_id,
    init_noise=None,
    physics_init_noise=None,
    prefix_actions=None,
):
    """Capture full VLA forward as a single CUDA graph.

    Args:
        gs_vla: GraphSafeVLA instance
        vl_input: sample VLM input dict
        state: (B, 1, state_dim) sample state
        embodiment_id: (B,) sample embodiment IDs
        init_noise: (B, action_horizon, action_dim) or None
        physics_init_noise: (B, fut_len, physics_dim) or None
        prefix_actions: (B, prefix_len, action_dim) for RTC trained mode,
            or None when ``gs_action_model.prefix_len == 0``.

    Returns:
        (replay_fn, static_output) — ``replay_fn(vl_input, state,
        embodiment_id, init_noise=, physics_init_noise=, prefix_actions=)``.
    """
    # Static input buffers
    static_vl_input = {}
    for k, v in vl_input.items():
        if isinstance(v, torch.Tensor):
            t = v.clone()
            if k == "pixel_values" and t.ndim == 3:
                t = t.reshape(-1, t.shape[-1])
            static_vl_input[k] = t
        else:
            static_vl_input[k] = v

    static_state = state.clone()
    static_embodiment_id = embodiment_id.clone()
    static_init_noise = init_noise.clone() if init_noise is not None else None
    static_physics_init_noise = (
        physics_init_noise.clone() if physics_init_noise is not None else None
    )
    static_prefix_actions = prefix_actions.clone() if prefix_actions is not None else None

    # Warmup in side stream
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        gs_vla(
            static_vl_input,
            static_state,
            static_embodiment_id,
            init_noise=static_init_noise,
            physics_init_noise=static_physics_init_noise,
            prefix_actions=static_prefix_actions,
        )
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # CUDA Graph capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.no_grad():
        graph_output = gs_vla(
            static_vl_input,
            static_state,
            static_embodiment_id,
            init_noise=static_init_noise,
            physics_init_noise=static_physics_init_noise,
            prefix_actions=static_prefix_actions,
        )
    torch.cuda.synchronize()

    static_output = graph_output
    _print("  CUDA graph captured (full VLA pipeline)")

    # Replay determinism check
    graph.replay()
    torch.cuda.synchronize()
    r1 = static_output.clone()
    graph.replay()
    torch.cuda.synchronize()
    r2 = static_output.clone()
    _print(f"  Replay determinism: max_diff={(r1 - r2).abs().max().item():.6f}")

    def replay_fn(
        vl_input_,
        state_,
        embodiment_id_,
        init_noise=None,
        physics_init_noise=None,
        prefix_actions=None,
    ):
        for k, v in vl_input_.items():
            if k in static_vl_input and isinstance(v, torch.Tensor):
                t = v
                if k == "pixel_values" and t.ndim == 3:
                    t = t.reshape(-1, t.shape[-1])
                static_vl_input[k].copy_(t)
        static_state.copy_(state_)
        static_embodiment_id.copy_(embodiment_id_)
        if init_noise is not None and static_init_noise is not None:
            static_init_noise.copy_(init_noise)
        if physics_init_noise is not None and static_physics_init_noise is not None:
            static_physics_init_noise.copy_(physics_init_noise)
        if prefix_actions is not None and static_prefix_actions is not None:
            static_prefix_actions.copy_(prefix_actions)

        graph.replay()
        return static_output.clone()

    return replay_fn, static_output
