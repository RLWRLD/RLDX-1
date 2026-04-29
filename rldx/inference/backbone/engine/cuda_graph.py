"""Path B: CUDA Graph capture for GraphSafe backbone.

With GraphSafe wrappers applied, the entire backbone forward is graph-safe
(no .item(), .tolist(), torch.nonzero, etc.).  We simply capture
backbone(vl_input) as a single CUDA graph.
"""

from __future__ import annotations

import torch

from rldx.utils.dist import rank_zero_print as _print


def setup_cuda_graph_full(backbone, vl_input):
    """Capture backbone(vl_input) as a single CUDA graph.

    Requires GraphSafe wrappers to be already applied on the backbone.

    Returns:
        (graph, replay_forward, orig_backbone_forward)
    """
    # --- Static input buffers ---
    static_vl_input = {}
    for k, v in vl_input.items():
        if isinstance(v, torch.Tensor):
            t = v.clone()
            if k == "pixel_values" and t.ndim == 3:
                t = t.reshape(-1, t.shape[-1])
            if k == "image_grid_thw" and t.ndim == 3:
                t = t.reshape(-1, 3)
            static_vl_input[k] = t
        else:
            static_vl_input[k] = v

    # --- Warmup in side stream ---
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        backbone(static_vl_input)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # --- CUDA Graph capture ---
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.no_grad():
        graph_output = backbone(static_vl_input)
    torch.cuda.synchronize()

    static_features = graph_output["backbone_features"]
    _print("  CUDA graph captured (full backbone)")

    # --- Replay determinism check ---
    graph.replay()
    torch.cuda.synchronize()
    r1 = static_features.clone()
    graph.replay()
    torch.cuda.synchronize()
    r2 = static_features.clone()
    _print(f"  Replay determinism check: max_diff={(r1 - r2).abs().max().item():.6f}")

    # --- Replay forward ---
    orig_backbone_forward = backbone.forward

    def replay_forward(vl_input_):
        # Copy new inputs into static buffers
        for k, v in vl_input_.items():
            if k in static_vl_input and isinstance(v, torch.Tensor):
                t = v
                if k == "pixel_values" and t.ndim == 3:
                    t = t.reshape(-1, t.shape[-1])
                if k == "image_grid_thw" and t.ndim == 3:
                    t = t.reshape(-1, 3)
                static_vl_input[k].copy_(t)

        graph.replay()

        return {
            "backbone_features": static_features.clone(),
            "backbone_attention_mask": vl_input_.get("attention_mask"),
            "image_mask": None,
        }

    return graph, replay_forward, orig_backbone_forward
