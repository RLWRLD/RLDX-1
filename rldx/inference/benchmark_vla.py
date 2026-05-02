"""
Full VLA (Vision-Language-Action) Pipeline Benchmark.

End-to-end: backbone → vlln → ActionModel → action trajectory.

Benchmark paths (always run in order):
  A: Vanilla                        — Original PyTorch model, eager execution (baseline)
  B: Torch Inductor (vanilla)       — torch.compile on vanilla sub-modules (compiler only)
  C: GraphSafe + CUDA Graph         — GraphSafe wrapping + CUDA Graph capture
  D: Custom Chain                   — GraphSafe + custom Triton kernels + torch.compile

Usage:
  python inference/benchmark_vla.py
  python inference/benchmark_vla.py --iter 100 --warmup 50
"""

import argparse
import os
import sys
import time as _time
import traceback

import torch
import torch._inductor.config as _inductor_config

# Patch: allow complex64 dtype in safetensors loading
from transformers.modeling_utils import str_to_torch_dtype


if "C64" not in str_to_torch_dtype:
    str_to_torch_dtype["C64"] = torch.complex64

_inductor_config.max_autotune_gemm_backends = "ATEN"
_inductor_config.emulate_precision_casts = True

# Path setup
import _path  # noqa: E402


_path.setup(__file__)
# Sub-module internal imports need their dirs in path,
# but AFTER inference/ so that `model` still resolves to inference/model/.
for _subdir in ("action_model", "backbone", "memory"):
    _d = os.path.join(_path.INFERENCE_DIR, _subdir)
    if _d not in sys.path:
        sys.path.append(_d)

# Imports (all fully-qualified via inference/)
from engine.cuda_graph import setup_vla_cuda_graph  # noqa: E402
from engine.custom_vla_chain import build_custom_vla_chain, compile_custom_vla_chain  # noqa: E402
from model.graph_safe_vla import GraphSafeVLA  # noqa: E402
from utils.registry import is_placeholder_path  # noqa: E402

from utils import (  # noqa: E402
    MODEL_REGISTRY,
    generate_synthetic_input,
    load_vla,
    measure_times,
    print_correctness,
    print_latency_table,
)


# CLI


def _resolve_image_hw(args):
    """Return (H, W) from --image-height/--image-width, or square --image-size."""
    h, w = args.image_height, args.image_width
    if (h is None) != (w is None):
        raise ValueError("--image-height and --image-width must be set together")
    return (h, w) if h is not None else (args.image_size, args.image_size)


def parse_args():
    parser = argparse.ArgumentParser(description="VLA Full Pipeline Benchmark")
    parser.add_argument(
        "--model-type", default="rldx_1_pretrain", choices=list(MODEL_REGISTRY.keys())
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--num-images", type=int, default=2, help="Number of camera views (V)")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=4,
        help="Number of temporal frames per view (T). Produces V*T images in time-major order.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square image side length. Ignored if both --image-height and --image-width are set.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=None,
        help="Image H for rectangular input. Must be set with --image-width.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=None,
        help="Image W for rectangular input. Must be set with --image-height.",
    )
    parser.add_argument("--concat-frames", action="store_true")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--n-state", type=int, default=1)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--iter", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-cog-tokens", action="store_true")
    parser.add_argument("--n-cog-tokens", type=int, default=64)
    parser.add_argument(
        "--compile-mode",
        default="max-autotune",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode for Path C/D. Path B fixes its own mode internally.",
    )
    # RTC inference-time controls (mirrors ``run_rldx_server.py``).  When
    # set, override the loaded model's ``rtc_inference_*`` config fields
    # at construction time so the bench can exercise both ``none`` and
    # ``trained`` modes from the CLI.  ``trained`` is gated on the
    # checkpoint's ``rtc_training_max_delay > 0`` (RTC paper Alg. 1) and
    # raises a clear ``ValueError`` from
    # ``rldx/model/modules/action_model/rtc.py::RTCConfig.validate`` when
    # not satisfied.
    # ``guided`` mode is intentionally excluded — it needs autograd +
    # per-step VJP support that the inference port doesn't carry yet
    # (see the original Isaac → RLDX port plan).  The dispatcher / chain
    # only support ``none`` and ``trained`` today; gate at the CLI rather
    # than letting it fall through to a partially-wired path.
    parser.add_argument(
        "--rtc-inference-mode",
        type=str,
        default=None,
        choices=[None, "none", "trained"],
        help="Override config.rtc_inference_mode (None = "
        "use checkpoint's value; default falls back "
        "to 'none').",
    )
    parser.add_argument(
        "--rtc-inference-delay",
        type=int,
        default=None,
        help="Override config.rtc_inference_delay (frozen prefix length d).",
    )
    return parser.parse_args()


# Build GraphSafe models


def build_graph_safe_vla(backbone, action_model, vl_input, args, meta, device, dtype):
    """Construct GraphSafeVLA from loaded backbone + action_model + optional memory."""
    from action_model.model import GraphSafeActionModel
    from backbone.model import GraphSafeQwen3VLBackbone

    gs_backbone = GraphSafeQwen3VLBackbone(
        backbone,
        vl_input,
        num_frames=args.num_frames,
        num_views=args.num_images,
    )

    # Derive n_vl from VLM output shape (not hardcoded)
    with torch.no_grad():
        vl_out = gs_backbone(vl_input)
    n_vl_raw = vl_out.shape[1]
    print(f"  VLM output: {list(vl_out.shape)} → n_vl_raw={n_vl_raw}")

    # Optional memory module
    gs_memory = None
    memory_config = meta.get("memory_config")
    if meta.get("memory_module") is not None and memory_config is not None:
        from memory.model import GraphSafeMemory

        gs_memory = GraphSafeMemory(
            memory_module=meta["memory_module"],
            memory_length=memory_config["memory_length"],
            memory_n_cog_tokens=memory_config["memory_n_cog_tokens"],
            device=device,
            dtype=dtype,
        ).eval()

        # Run memory once to determine output n_vl
        n_q = memory_config["n_cog_tokens"]
        n_cog_mem = memory_config["memory_n_cog_tokens"]
        K = memory_config["memory_length"]
        cog_current = vl_out[:, (n_q - n_cog_mem) :, :]
        dummy_cache = cog_current.repeat(1, K, 1)
        with torch.no_grad():
            gs_memory(dummy_cache)

        if memory_config["concat_memory"]:
            n_vl = n_vl_raw + n_cog_mem
        else:
            n_vl = n_vl_raw
        print(
            f"  Memory: n_vl_raw={n_vl_raw} → n_vl={n_vl} "
            f"(concat={memory_config['concat_memory']}, +{n_cog_mem} augmented)"
        )
    else:
        n_vl = n_vl_raw

    N_sa = args.n_state + args.action_horizon

    gs_action_model = GraphSafeActionModel(
        action_model=action_model,
        n_vl=n_vl,
        n_sa_pure=N_sa,
        action_horizon=args.action_horizon,
        action_dim=action_model.action_decoder.layer2.W.shape[2],
        num_inference_timesteps=args.denoising_steps,
        device=device,
        dtype=dtype,
    ).eval()

    gs_vla = GraphSafeVLA(
        gs_backbone, gs_action_model, gs_memory=gs_memory, memory_config=memory_config
    ).eval()
    return gs_vla


# Main


def main():
    args = parse_args()
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    dtype = torch.bfloat16
    B = 1

    # ---- Load full model ----
    print("=" * 80)
    print("VLA Full Pipeline Benchmark")
    print("=" * 80)

    backbone, action_model, meta = load_vla(
        model_type=args.model_type,
        model_path=args.model_path,
        device=args.device,
        rtc_inference_mode=args.rtc_inference_mode,
        rtc_inference_delay=args.rtc_inference_delay,
    )
    full_model = meta["full_model"]

    # ---- Generate synthetic input ----
    # ``processor_path`` resolution order:
    #   1. ``--model-path`` if given (always wins for local-dir loads)
    #   2. registry's ``processor_path`` if it isn't still a release-time
    #      ``TBD_*`` placeholder (see ``utils.registry.is_placeholder_path``)
    #   3. registry's ``hf_path`` as a last resort
    model_cfg = meta["model_cfg"]
    registry_proc = model_cfg.get("processor_path")
    if args.model_path:
        processor_path = args.model_path
    elif registry_proc and not is_placeholder_path(registry_proc):
        processor_path = registry_proc
    else:
        processor_path = model_cfg["hf_path"]
    image_h, image_w = _resolve_image_hw(args)

    print("\nGenerating input...")
    vl_input, input_info = generate_synthetic_input(
        processor_path,
        args.num_images,
        image_h,
        image_w,
        args.concat_frames,
        None,
        device,
        args.seed,
        custom_prompt=args.prompt,
        num_frames=args.num_frames,
    )
    print(f'  prompt: "{input_info.get("prompt_text", "?")}"')
    print(f"  views={args.num_images}, frames={args.num_frames}")
    print(
        f"  seq_len={input_info.get('seq_len', '?')} "
        f"(vision={input_info.get('vision_tokens', '?')}, "
        f"prompt={input_info.get('prompt_tokens', '?')})"
    )

    # Action model inputs
    torch.manual_seed(args.seed)
    max_state_dim = action_model.state_encoder.layer1.W.shape[1]
    state = torch.randn((B, args.n_state, max_state_dim), device=device, dtype=dtype) * 0.01
    embodiment_id = torch.zeros(B, dtype=torch.long, device=device)

    # Vanilla model input: merge VLM input + action head inputs
    vanilla_input = dict(vl_input)
    vanilla_input["state"] = state
    vanilla_input["embodiment_id"] = embodiment_id

    # ---- Pre-generate noise (same order as vanilla model's internal torch.randn) ----
    # Vanilla model generates: [physics_fut noise (if physics)] → [action noise]
    # We generate in the same order so the random state matches.
    use_physics = meta.get("use_physics", False)
    action_dim = action_model.action_decoder.layer2.W.shape[2]

    torch.manual_seed(args.seed + 1)
    if use_physics:
        physics = action_model.physics
        physics_fut_len = physics.physics_fut_len
        physics_dim = physics.physics_dim
        physics_init_noise = torch.randn(
            (B, physics_fut_len, physics_dim), device=device, dtype=dtype
        )
    else:
        physics_init_noise = None
    init_noise = torch.randn((B, args.action_horizon, action_dim), device=device, dtype=dtype)

    # =========================================================================
    # Helpers
    # =========================================================================
    results = {}
    outputs = {}
    build_times = {}

    def run_benchmark(label, fn):
        with torch.no_grad():
            output = fn()
        # Clone immediately — torch.compile(max-autotune) uses CUDA graphs
        # internally, so subsequent calls overwrite the output tensor's memory
        output = output.detach().clone()
        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f"  [!] {label}: NaN/Inf detected")
        print(f"  Warming up ({args.warmup} iters)...")
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        print(f"  Benchmarking ({args.iter} iters)...")
        times = measure_times(fn, args.iter)
        results[label] = times
        outputs[label] = output

    def make_fn(forward_fn):
        def fn():
            with torch.no_grad():
                return forward_fn(
                    vl_input,
                    state,
                    embodiment_id,
                    init_noise=init_noise,
                    physics_init_noise=physics_init_noise,
                )

        return fn

    # =========================================================================
    # Path A: Vanilla RLDX (eager)
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Path A: Vanilla RLDX (eager)")
    print(f"{'=' * 60}")

    # Seed so internal torch.randn() produces same noise as init_noise
    torch.manual_seed(args.seed + 1)

    def vanilla_fn():
        with torch.no_grad():
            return full_model.get_action(vanilla_input)["action_pred"]

    run_benchmark("A: Vanilla", vanilla_fn)

    vanilla_action = outputs["A: Vanilla"]
    print(f"  Output shape: {list(vanilla_action.shape)} {vanilla_action.dtype}")
    print(f"  Output range: [{vanilla_action.min().item():.4f}, {vanilla_action.max().item():.4f}]")

    # =========================================================================
    # Path B: Torch Inductor on Vanilla (sub-module compile)
    # =========================================================================
    # Per-leaf compile produces one CUDA Graph per leaf, and cudagraph_trees
    # cannot track buffer ownership across the eager Python glue between them.
    # ``max-autotune-no-cudagraphs`` keeps Triton autotune while dropping the
    # CUDA Graph wrap. Vision tower stays eager — flash-attn varlen op is
    # Dynamo-incompatible (FakeTensor rejected by SymInt schema arg).
    _compile_mode = "max-autotune-no-cudagraphs"
    print(f"\n{'=' * 60}")
    print(f"Path B: Torch Inductor on Vanilla (sub-module, {_compile_mode})")
    print(f"{'=' * 60}")

    llm = full_model.backbone.qwen_model.model.language_model
    action_model_ref = full_model.action_model
    memory_attr = "memory" if hasattr(full_model, "memory") else "_memory_module"

    orig_llm_layers: list = []
    orig_action_enc = action_model_ref.action_encoder
    orig_state_enc = action_model_ref.state_encoder
    orig_msat = action_model_ref.model
    orig_action_dec = getattr(action_model_ref, "action_decoder", None)
    orig_physics = getattr(action_model_ref, "physics", None)
    orig_memory = getattr(full_model, memory_attr, None)

    try:
        torch._dynamo.reset()
        compiled_modules: list[str] = []
        t0 = _time.time()

        for i, layer in enumerate(llm.layers):
            inner = layer.layer if hasattr(layer, "layer") else layer
            orig_llm_layers.append(inner)
            compiled = torch.compile(inner, mode=_compile_mode)
            if hasattr(layer, "layer"):
                layer.layer = compiled
            else:
                llm.layers[i] = compiled
        compiled_modules.append(f"LLM×{len(llm.layers)}")

        action_model_ref.action_encoder = torch.compile(
            action_model_ref.action_encoder, mode=_compile_mode
        )
        compiled_modules.append("action_encoder")

        action_model_ref.state_encoder = torch.compile(
            action_model_ref.state_encoder, mode=_compile_mode
        )
        compiled_modules.append("state_encoder")

        action_model_ref.model = torch.compile(action_model_ref.model, mode=_compile_mode)
        compiled_modules.append("MSAT")

        if orig_action_dec is not None:
            action_model_ref.action_decoder = torch.compile(
                action_model_ref.action_decoder, mode=_compile_mode
            )
            compiled_modules.append("action_decoder")

        if orig_physics is not None:
            action_model_ref.physics = torch.compile(action_model_ref.physics, mode=_compile_mode)
            compiled_modules.append("physics")

        if orig_memory is not None:
            setattr(full_model, memory_attr, torch.compile(orig_memory, mode=_compile_mode))
            compiled_modules.append("memory")

        print(f"  Compiled: {', '.join(compiled_modules)}")

        torch.manual_seed(args.seed + 1)
        with torch.no_grad():
            full_model.get_action(vanilla_input)
        torch.cuda.synchronize()
        build_times["B: Inductor"] = _time.time() - t0
        print(f"  Compilation: {build_times['B: Inductor']:.1f}s")

        torch.manual_seed(args.seed + 1)

        def inductor_fn():
            with torch.no_grad():
                return full_model.get_action(vanilla_input)["action_pred"]

        run_benchmark("B: Inductor (vanilla)", inductor_fn)
    except Exception as e:
        print(f"  [Inductor] Failed: {e}")
        traceback.print_exc()
    finally:
        for i, layer in enumerate(llm.layers):
            if i < len(orig_llm_layers):
                if hasattr(layer, "layer"):
                    layer.layer = orig_llm_layers[i]
                else:
                    llm.layers[i] = orig_llm_layers[i]
        action_model_ref.action_encoder = orig_action_enc
        action_model_ref.state_encoder = orig_state_enc
        action_model_ref.model = orig_msat
        if orig_action_dec is not None:
            action_model_ref.action_decoder = orig_action_dec
        if orig_physics is not None:
            action_model_ref.physics = orig_physics
        if orig_memory is not None:
            setattr(full_model, memory_attr, orig_memory)
        torch._dynamo.reset()

    # =========================================================================
    # Path C: GraphSafe + CUDA Graph
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Path C: GraphSafe + CUDA Graph")
    print(f"{'=' * 60}")
    try:
        print("  Building GraphSafeVLA...")
        gs_vla = build_graph_safe_vla(backbone, action_model, vl_input, args, meta, device, dtype)

        replay_fn, _ = setup_vla_cuda_graph(
            gs_vla,
            vl_input,
            state,
            embodiment_id,
            init_noise=init_noise,
            physics_init_noise=physics_init_noise,
        )

        def cg_forward(vl_in, st, emb, init_noise=None, physics_init_noise=None):
            return replay_fn(
                vl_in, st, emb, init_noise=init_noise, physics_init_noise=physics_init_noise
            )

        run_benchmark("C: GraphSafe + CG", make_fn(cg_forward))
    except Exception as e:
        print(f"  [GraphSafe + CG] Failed: {e}")
        traceback.print_exc()

    # =========================================================================
    # Path D: CustomVLAChain (Custom Triton kernels + torch.compile)
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Path D: CustomVLAChain (Custom Triton kernels + torch.compile)")
    print(f"{'=' * 60}")
    try:
        # gs_vla must exist from Path D
        if "gs_vla" not in locals():
            print("  Building GraphSafeVLA...")
            gs_vla = build_graph_safe_vla(
                backbone, action_model, vl_input, args, meta, device, dtype
            )

        # 1. Build unified chain (VLM + ActionModel)
        vla_chain = build_custom_vla_chain(gs_vla, device, dtype)

        # 2. Prepare sample inputs for compilation trigger
        pv = vl_input["pixel_values"]
        if pv.ndim == 3:
            pv = pv.reshape(-1, pv.shape[-1])
        pv = pv.type(gs_vla.gs_backbone.gs_visual.dtype)

        # Reuse global init_noise (same as all other paths)
        sample_inputs = (pv, state, embodiment_id, init_noise)

        # 3. Compile as single unit
        compiled_vla, compile_time = compile_custom_vla_chain(
            vla_chain,
            sample_inputs,
            compile_mode=args.compile_mode,
        )
        build_times["D: CustomVLA"] = compile_time

        # 4. Benchmark — reuse the same pv tensor (CUDA graph expects same address)
        def custom_vla_fn(vl_in, st, emb, init_noise=None, physics_init_noise=None):
            return compiled_vla(pv, st, emb, init_noise=init_noise)

        run_benchmark("D: CustomVLAChain", make_fn(custom_vla_fn))

        torch._dynamo.reset()
    except Exception as e:
        print(f"  [CustomVLAChain] Failed: {e}")
        traceback.print_exc()

    # =========================================================================
    # Report
    # =========================================================================
    print()
    print("=" * 80)
    print("VLA Full Pipeline Benchmark")
    print("=" * 80)
    print(f"Model: {args.model_type}")
    print(
        f"VLM: hidden={meta['hidden_size']}, llm_layers={meta['num_llm_layers']}, "
        f"select_layer={meta['select_layer']}"
    )
    msat = action_model.model
    print(
        f"ActionModel: sa_dim={msat.inner_dim}, vl_dim={msat.vl_proj_to_sa.in_features}, "
        f"action_dim={action_dim}"
    )
    print(f"Pipeline: VLM → vlln → {args.denoising_steps}x[action_enc + MSAT + action_dec + euler]")
    size_str = f"{image_h}x{image_w}" if image_h != image_w else f"{image_h}"
    print(f"Input: images={args.num_images}x{size_str}, seq_len={input_info.get('seq_len', '?')}")
    print(f"Iter={args.iter}, warmup={args.warmup}")
    if build_times:
        print(f"Build: {', '.join(f'{k}={v:.1f}s' for k, v in build_times.items())}")

    print_latency_table(
        f"Full VLA Pipeline ({args.denoising_steps} denoising steps)",
        results,
    )

    if "A: Vanilla" in outputs and len(outputs) > 1:
        corr_entries = [
            (label, out, None) for label, out in outputs.items() if label != "A: Vanilla"
        ]
        print_correctness("Correctness (vs A: Vanilla)", corr_entries, vanilla_action)

    print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated(device) / (1024**2):.1f} MB\n")


if __name__ == "__main__":
    main()
