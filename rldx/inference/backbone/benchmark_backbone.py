"""
Backbone Benchmark: RLDX v1.4-v / v1.4-all

Full pipeline benchmark (vision + rope + LLM + projection).

Benchmark paths (always run in order):
  A: Vanilla                           — Original PyTorch model, eager execution (baseline)
  B: Torch Inductor (vanilla)          — torch.compile on vanilla LLM layers (compiler only)
  C: GraphSafe + CUDA Graph            — GraphSafe wrapping + CUDA Graph capture
  D: Custom Chain                      — GraphSafe + custom Triton kernels + torch.compile

Usage:
  python inference/backbone/benchmark_backbone.py
  python inference/backbone/benchmark_backbone.py --mode all
"""

import argparse
import ctypes
import os
import sys
import time as _time


# Fix NVRTC builtins path
try:
    import nvidia.cu13 as _cu13

    _cu13_lib = os.path.join(os.path.dirname(os.path.abspath(_cu13.__path__[0])), "cu13", "lib")
    if os.path.isdir(_cu13_lib):
        _ld = os.environ.get("LD_LIBRARY_PATH", "")
        if _cu13_lib not in _ld:
            os.environ["LD_LIBRARY_PATH"] = f"{_cu13_lib}:{_ld}" if _ld else _cu13_lib
        _builtins = os.path.join(_cu13_lib, "libnvrtc-builtins.so.13.0")
        if os.path.isfile(_builtins):
            ctypes.CDLL(_builtins)
except (ImportError, OSError):
    pass

import traceback

import torch
import torch._inductor.config as _inductor_config

# Patch: allow complex64 dtype in safetensors loading (inv_freq buffers)
from transformers.modeling_utils import str_to_torch_dtype


if "C64" not in str_to_torch_dtype:
    str_to_torch_dtype["C64"] = torch.complex64

_inductor_config.max_autotune_gemm_backends = "ATEN"
_inductor_config.emulate_precision_casts = True

# Path setup
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import _path  # noqa: E402


_path.setup(__file__)

# Imports
from engine import setup_cuda_graph_full  # noqa: E402

from model import GraphSafeQwen3VLBackbone, patch_backbone  # noqa: E402
from utils import (  # noqa: E402
    generate_synthetic_input,
    load_backbone,
    measure_times,
    print_correctness,
    print_latency_table,
)


# CLI

_MODE_TO_MODEL_TYPE = {
    "video": "rldx_1_pretrain",
    "all": "rldx_1_midtrain_allex",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Backbone Benchmark")
    parser.add_argument("--mode", default="video", choices=list(_MODE_TO_MODEL_TYPE.keys()))
    parser.add_argument("--num-images", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--concat-frames", action="store_true")
    parser.add_argument("--prompt", type=str, default=None)
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
        help="torch.compile mode for full-backbone paths. Path B fixes its own mode.",
    )
    args = parser.parse_args()
    args.model_type = _MODE_TO_MODEL_TYPE[args.mode]
    args.model_path = None
    return args


# Main


def main():
    args = parse_args()
    backbone, meta = load_backbone(args)
    device = meta["device"]

    processor_path = meta["model_cfg"].get(
        "processor_path", args.model_path or meta["model_cfg"]["hf_path"]
    )
    print("\nGenerating input...")
    vl_input, input_info = generate_synthetic_input(
        processor_path,
        args.num_images,
        args.image_size,
        args.image_size,
        args.concat_frames,
        None,
        device,
        args.seed,
        custom_prompt=args.prompt,
    )
    meta["input_info"] = input_info
    print(f'  prompt: "{input_info.get("prompt_text", "?")}"')
    print(
        f"  seq_len={input_info.get('seq_len', '?')} "
        f"(vision={input_info.get('vision_tokens', '?')}, "
        f"prompt_tokens={input_info.get('prompt_tokens', '?')})"
    )

    # =========================================================================
    # Helpers
    # =========================================================================
    results = {}
    outputs = {}
    build_times = {}

    def run_benchmark(label, fn):
        with torch.no_grad():
            output = fn()
        output = (
            output.detach().clone()
            if isinstance(output, torch.Tensor)
            else output["backbone_features"].detach().clone()
        )
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

    # =========================================================================
    # Path A: Vanilla (backbone, eager)
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Path A: Vanilla (backbone, eager)")
    print(f"{'=' * 60}")

    def vanilla_fn():
        with torch.no_grad():
            return backbone(vl_input)["backbone_features"]

    run_benchmark("A: Vanilla", vanilla_fn)
    vanilla_features = outputs["A: Vanilla"]
    print(f"  Output shape: {list(vanilla_features.shape)} {vanilla_features.dtype}")

    # =========================================================================
    # Path B: Torch Inductor on Vanilla (LLM sub-module compile)
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Path B: Torch Inductor on Vanilla (sub-module)")
    print(f"{'=' * 60}")

    from engine.torch_inductor import restore_submodule_compile, setup_submodule_compile

    try:
        torch._dynamo.reset()
        t0 = _time.time()
        orig_layers, compile_time = setup_submodule_compile(backbone, vl_input)
        build_times["B: Inductor"] = _time.time() - t0
        print(f"  Compilation: {build_times['B: Inductor']:.1f}s")

        def inductor_fn():
            with torch.no_grad():
                return backbone(vl_input)["backbone_features"]

        run_benchmark("B: Inductor (vanilla)", inductor_fn)
    except Exception as e:
        print(f"  [Inductor] Failed: {e}")
        traceback.print_exc()
    finally:
        if "orig_layers" in locals() and orig_layers is not None:
            restore_submodule_compile(backbone, orig_layers)
        torch._dynamo.reset()

    # =========================================================================
    # Build GraphSafe Backbone (needed for Paths C and D)
    # =========================================================================
    num_frames = input_info.get("num_images", 1) if args.concat_frames else 1
    num_views = 1
    gs_backbone = GraphSafeQwen3VLBackbone(
        backbone, vl_input, num_frames=num_frames, num_views=num_views
    )
    patch_backbone(backbone, gs_backbone)

    # =========================================================================
    # Path C: GraphSafe + CUDA Graph
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Path C: GraphSafe + CUDA Graph")
    print(f"{'=' * 60}")
    try:
        _, replay_forward, orig_forward = setup_cuda_graph_full(backbone, vl_input)
        backbone.forward = replay_forward

        def cg_fn():
            with torch.no_grad():
                return backbone(vl_input)["backbone_features"]

        run_benchmark("C: GraphSafe + CG", cg_fn)
        backbone.forward = orig_forward
    except Exception as e:
        print(f"  [GraphSafe + CG] Failed: {e}")
        traceback.print_exc()

    # =========================================================================
    # Path D: CustomVLMChain + torch.compile
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Path D: CustomVLMChain + torch.compile")
    print(f"{'=' * 60}")
    try:
        from engine.custom_backbone_chain import (
            build_custom_backbone_chain,
            compile_custom_backbone_chain,
        )

        pv = vl_input["pixel_values"]
        if pv.ndim == 3:
            pv = pv.reshape(-1, pv.shape[-1])
        pv = pv.type(gs_backbone.gs_visual.dtype)

        print("  Building CustomVLMChain...")
        backbone_chain = build_custom_backbone_chain(gs_backbone)
        compiled_chain, chain_compile_time = compile_custom_backbone_chain(backbone_chain, pv)
        build_times["D: CustomVLM"] = chain_compile_time

        def custom_fn():
            with torch.no_grad():
                return compiled_chain(pv)

        run_benchmark("D: CustomVLMChain", custom_fn)
        torch._dynamo.reset()
    except Exception as e:
        print(f"  [CustomVLMChain] Failed: {e}")
        traceback.print_exc()

    # =========================================================================
    # Report
    # =========================================================================
    print()
    print("=" * 80)
    print(f"Backbone Benchmark: {args.model_type}")
    print("=" * 80)
    hidden_size = meta["hidden_size"]
    num_llm_layers = meta["num_llm_layers"]
    print(
        f"Config: hidden={hidden_size}, llm_layers={num_llm_layers}, "
        f"select_layers={backbone.select_layers}"
    )
    print(
        f"Input:  images={args.num_images}x{args.image_size}, "
        f"seq_len={input_info.get('seq_len', '?')}"
    )
    print(f"Iter={args.iter}, warmup={args.warmup}")
    if build_times:
        print(f"Build: {', '.join(f'{k}={v:.1f}s' for k, v in build_times.items())}")

    print_latency_table("Full Backbone Pipeline", results)

    if "A: Vanilla" in outputs and len(outputs) > 1:
        corr_entries = [(label, out) for label, out in outputs.items() if label != "A: Vanilla"]
        print_correctness("Correctness (vs A: Vanilla)", corr_entries, vanilla_features)

    print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated(device) / (1024**2):.1f} MB\n")


if __name__ == "__main__":
    main()
