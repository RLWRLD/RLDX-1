"""
Memory Module (TransformerMemory) Benchmark.

Standalone benchmark for the Transformer-based memory module that aggregates
cog-token embeddings across K timesteps.

Input shape:  (B, K * n_cog_mem, hidden_size) = (1, 64, 1536) for default config
Output shape: (B, K * n_cog_mem, hidden_size) = (1, 64, 1536)

Benchmark paths (always run in order):
  A: Vanilla                           — Original PyTorch model, eager execution (baseline)
  B: Torch Inductor (vanilla)          — torch.compile on vanilla module (compiler only)
  C: GraphSafe + CUDA Graph            — GraphSafe wrapping + CUDA Graph capture
  D: Custom Chain                      — GraphSafe + custom Triton kernels + torch.compile

Usage:
  python inference/memory/benchmark_memory.py
  python inference/memory/benchmark_memory.py --iter 500 --warmup 200
"""

import argparse
import os
import sys
import time as _time
import traceback

import torch
import torch._inductor.config


torch._inductor.config.triton.cudagraph_trees = False

# Path setup
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import _path  # noqa: E402


_path.setup(__file__)

# Imports
from engine import (  # noqa: E402
    build_custom_memory_chain,
    compile_custom_memory_chain,
    setup_cuda_graph,
)

from model import GraphSafeMemory  # noqa: E402
from utils import load_memory, measure_times, print_correctness, print_latency_table  # noqa: E402


# Main


def main():
    parser = argparse.ArgumentParser(description="Memory Module (TransformerMemory) benchmark")
    parser.add_argument("--model-type", type=str, default="rldx_1_midtrain_allex")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--iter", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--compile-mode",
        default="max-autotune",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)

    B = 1

    # =========================================================================
    # Load memory module from checkpoint
    # =========================================================================
    print("=" * 80)
    print("Memory Module (TransformerMemory) Benchmark")
    print("=" * 80)

    memory_module, mem_cfg = load_memory(
        model_type=args.model_type,
        model_path=args.model_path,
        device=args.device,
    )

    K = mem_cfg["memory_length"]
    n_cog_mem = mem_cfg["memory_n_cog_tokens"]
    hidden_size = mem_cfg["hidden_size"]
    seq_length = K * n_cog_mem

    n_params = sum(p.numel() for p in memory_module.parameters()) / 1e6
    print(f"\n  seq_length={seq_length} (K={K} x n_cog_mem={n_cog_mem})")
    print(
        f"  hidden_size={hidden_size}, layers={mem_cfg['num_layers']}, heads={mem_cfg['num_heads']}"
    )
    print(
        f"  block_attn_size={mem_cfg['block_attn_size']}, "
        f"causal={mem_cfg['use_causal_attn']}, rope={mem_cfg['use_rope']}"
    )
    print(f"  Params: {n_params:.2f}M")

    # =========================================================================
    # Test input — simulates K timesteps of memory cog tokens
    # =========================================================================
    torch.manual_seed(args.seed)
    inputs_embeds = torch.randn((B, seq_length, hidden_size), device=device, dtype=dtype) * 0.01

    # =========================================================================
    # Helpers
    # =========================================================================
    results = {}
    outputs = {}
    build_times = {}

    def run_benchmark(label, fn):
        with torch.no_grad():
            output = fn()
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

    def make_fn(module):
        def fn():
            with torch.no_grad():
                return module(inputs_embeds)

        return fn

    # =========================================================================
    # Path A: Vanilla (TransformerMemory, eager)
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Path A: Vanilla (TransformerMemory, eager)")
    print(f"{'=' * 60}")
    run_benchmark("A: Vanilla", make_fn(memory_module))

    vanilla_out = outputs["A: Vanilla"]
    print(f"  Output shape: {list(vanilla_out.shape)} {vanilla_out.dtype}")
    print(f"  Output range: [{vanilla_out.min().item():.4f}, {vanilla_out.max().item():.4f}]")

    # =========================================================================
    # Path B: Torch Inductor on Vanilla
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f"Path B: Torch Inductor on Vanilla ({args.compile_mode})")
    print(f"{'=' * 60}")
    try:
        torch._dynamo.reset()
        compiled_vanilla = torch.compile(memory_module, mode=args.compile_mode)

        print("  Triggering compilation...")
        t0 = _time.time()
        with torch.no_grad():
            compiled_vanilla(inputs_embeds)
        torch.cuda.synchronize()
        build_times["B: Inductor"] = _time.time() - t0
        print(f"  Compilation: {build_times['B: Inductor']:.1f}s")

        run_benchmark("B: Inductor (vanilla)", make_fn(compiled_vanilla))
        torch._dynamo.reset()
    except Exception as e:
        print(f"  [Inductor] Failed: {e}")
        traceback.print_exc()

    # =========================================================================
    # Path C: GraphSafe + CUDA Graph
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Path C: GraphSafe + CUDA Graph")
    print(f"{'=' * 60}")
    try:
        print("  Building GraphSafeMemory...")
        gs_memory = GraphSafeMemory(
            memory_module=memory_module,
            memory_length=K,
            memory_n_cog_tokens=n_cog_mem,
            device=device,
            dtype=dtype,
        ).eval()

        cg_memory = setup_cuda_graph(gs_memory)

        print("  Capturing CUDA graph...")
        with torch.no_grad():
            cg_memory(inputs_embeds)
        torch.cuda.synchronize()
        print("  CUDA graph captured")

        run_benchmark("C: GraphSafe + CG", make_fn(cg_memory))
    except Exception as e:
        print(f"  [GraphSafe + CG] Failed: {e}")
        traceback.print_exc()

    # =========================================================================
    # Path D: CustomMemoryChain + torch.compile
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Path D: CustomMemoryChain + torch.compile")
    print(f"{'=' * 60}")
    try:
        if "gs_memory" not in locals():
            gs_memory = GraphSafeMemory(
                memory_module=memory_module,
                memory_length=K,
                memory_n_cog_tokens=n_cog_mem,
                device=device,
                dtype=dtype,
            ).eval()

        print("  Building CustomMemoryChain...")
        custom_chain = build_custom_memory_chain(gs_memory, device=device, dtype=dtype)

        compiled_chain, chain_compile_time = compile_custom_memory_chain(
            custom_chain, inputs_embeds, compile_mode=args.compile_mode
        )
        build_times["D: MemoryChain"] = chain_compile_time

        run_benchmark("D: CustomMemoryChain", make_fn(compiled_chain))
    except Exception as e:
        print(f"  [MemoryChain] Failed: {e}")
        traceback.print_exc()

    # =========================================================================
    # Report
    # =========================================================================
    print()
    print("=" * 80)
    print("Memory Module (TransformerMemory) Benchmark")
    print("=" * 80)
    print(
        f"Config: hidden={hidden_size}, layers={mem_cfg['num_layers']}, "
        f"heads={mem_cfg['num_heads']}, "
        f"block_attn_size={mem_cfg['block_attn_size']}"
    )
    print(
        f"Input:  (B={B}, seq={seq_length}, d={hidden_size}) — "
        f"K={K} timesteps x {n_cog_mem} cog tokens"
    )
    print(f"Params: {n_params:.2f}M")
    print(f"Iter={args.iter}, warmup={args.warmup}")
    if build_times:
        print(f"Build: {', '.join(f'{k}={v:.1f}s' for k, v in build_times.items())}")

    print_latency_table("Latency (single forward)", results)

    if "A: Vanilla" in outputs and len(outputs) > 1:
        corr_entries = [(label, out) for label, out in outputs.items() if label != "A: Vanilla"]
        print_correctness("Correctness (vs A: Vanilla)", corr_entries, vanilla_out)

    print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated(device) / (1024**2):.1f} MB\n")


if __name__ == "__main__":
    main()
