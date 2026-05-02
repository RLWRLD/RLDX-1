"""torch.compile optimization for backbone.

Two compilation strategies:
  1. setup_compile: full backbone.forward as single compiled unit
  2. setup_submodule_compile: individual LLM decoder layers compiled separately
     (fixed to ``max-autotune-no-cudagraphs`` — per-leaf CUDA Graphs collide
     on shared output buffers across sequential layer execution).

For custom Triton kernel chains, see custom_backbone_chain.py.
"""

from __future__ import annotations

import time as _time

import torch

from rldx.utils.dist import rank_zero_print as _print


_SUBMODULE_COMPILE_MODE = "max-autotune-no-cudagraphs"


def setup_compile(backbone, vl_input, select_layer=18, mode="max-autotune"):
    """Compile backbone.forward with torch.compile.

    Args:
        backbone: backbone model
        vl_input: sample input dict (triggers compilation)
        select_layer: controls dynamo cache_size_limit
        mode: torch.compile mode

    Returns:
        (compiled_forward, orig_forward, compile_time_s)
    """
    torch._dynamo.config.cache_size_limit = select_layer * 2
    torch._dynamo.reset()

    orig_forward = backbone.forward
    compiled_forward = torch.compile(backbone.forward, mode=mode)

    # Trigger compilation
    backbone.forward = compiled_forward
    t0 = _time.time()
    with torch.no_grad():
        backbone(vl_input)
    torch.cuda.synchronize()
    compile_time_s = _time.time() - t0
    _print(f"  Compilation: {compile_time_s:.1f}s")

    # Warmup (stabilize CUDA graph captures for reduce-overhead mode)
    for _ in range(3):
        with torch.no_grad():
            backbone(vl_input)
    torch.cuda.synchronize()

    backbone.forward = orig_forward
    return compiled_forward, orig_forward, compile_time_s


def setup_submodule_compile(backbone, vl_input):
    """Compile individual LLM decoder layers with torch.compile.

    Mode is fixed to ``max-autotune-no-cudagraphs``: each compiled layer
    becomes its own CUDA Graph, and cudagraph_trees cannot share buffers
    across the eager Python loop stitching them together.

    Args:
        backbone: backbone model (Qwen3VL-based)
        vl_input: sample input dict (triggers compilation)

    Returns:
        (orig_layers, compile_time_s)
    """
    llm = backbone.qwen_model.model.language_model

    torch._dynamo.reset()
    t0 = _time.time()

    orig_layers = []
    _print(f"  Compiling {len(llm.layers)} LLM layers (mode={_SUBMODULE_COMPILE_MODE})...")
    for i, layer in enumerate(llm.layers):
        inner = layer.layer if hasattr(layer, "layer") else layer
        orig_layers.append(inner)
        compiled = torch.compile(inner, mode=_SUBMODULE_COMPILE_MODE)
        if hasattr(layer, "layer"):
            layer.layer = compiled
        else:
            llm.layers[i] = compiled

    with torch.no_grad():
        backbone(vl_input)
    torch.cuda.synchronize()
    compile_time_s = _time.time() - t0
    _print(f"  Sub-module compilation: {compile_time_s:.1f}s")

    return orig_layers, compile_time_s


def restore_submodule_compile(backbone, orig_layers):
    """Restore original LLM layers after compiled benchmark."""
    llm = backbone.qwen_model.model.language_model
    for i, layer in enumerate(llm.layers):
        if i < len(orig_layers):
            if hasattr(layer, "layer"):
                layer.layer = orig_layers[i]
            else:
                llm.layers[i] = orig_layers[i]
    torch._dynamo.reset()


# Divergence analysis


def capture_with_hooks(language_model, backbone, vl_input, forward_fn):
    """Run forward pass and capture inputs_embeds + per-layer LLM outputs.

    Useful for comparing eager vs compiled per-layer divergence.

    Args:
        language_model: LLM text model (has _cached_inputs_embeds)
        backbone: backbone
        vl_input: input dict
        forward_fn: forward function (eager or compiled)

    Returns:
        (inputs_embeds, list_of_per_layer_outputs)
    """
    layer_store = []

    def _layer_hook(store):
        def hook(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            store.append(out.detach().clone())

        return hook

    inner_layers = []
    for layer in language_model.layers:
        inner = layer.layer if hasattr(layer, "layer") else layer
        inner_layers.append(inner)
    hooks = [layer.register_forward_hook(_layer_hook(layer_store)) for layer in inner_layers]

    saved_forward = backbone.forward
    backbone.forward = forward_fn
    with torch.no_grad():
        backbone(vl_input)
    backbone.forward = saved_forward

    for h in hooks:
        h.remove()

    ie = getattr(language_model, "_cached_inputs_embeds", None)
    return ie, layer_store
