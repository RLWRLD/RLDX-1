"""
Full Action Head (v1.4 MSAT) benchmark — supports v1.4-v and v1.4-all.

Benchmark paths (always run in order):
  A: Vanilla                           — Original PyTorch model, eager execution (baseline)
  B: Torch Inductor (vanilla)          — torch.compile on vanilla MSAT (compiler only)
  C: GraphSafe + CUDA Graph            — GraphSafe wrapping + CUDA Graph capture
  D: Custom Chain                      — GraphSafe + custom Triton kernels + torch.compile

Usage:
  python inference/action_model/benchmark_action_model.py                                  # v1.4-v (no physics)
  python inference/action_model/benchmark_action_model.py --model-type rldx_1_midtrain_allex  # v1.4-all (physics)
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

# Imports from package structure
from engine import setup_cuda_graph  # noqa: E402

from model import GraphSafeActionModel  # noqa: E402
from utils import (  # noqa: E402
    MODEL_REGISTRY,
    load_action_model,
    measure_times,
    print_correctness,
    print_latency_table,
)


# Vanilla action head runner (calls original sub-modules directly)


def vanilla_action_head_forward(
    components,
    vl_embs,
    state,
    embodiment_id,
    init_noise,
    action_horizon,
    denoising_steps,
    physics_hist=None,
    physics_init_noise=None,
):
    """Run the vanilla action-model pipeline using the original sub-modules.

    Mirrors RLDXActionModel.get_action() but accepts raw tensors.
    """
    am = components["action_model"]
    msat = components["msat"]
    B = vl_embs.shape[0]
    use_physics = components.get("use_physics", False)
    dt = 1.0 / denoising_steps

    vl = am.vlln(vl_embs)
    state_features = am.state_encoder(state, embodiment_id)
    pos_ids = torch.arange(action_horizon, device=vl_embs.device)
    pos_embs = am.position_embedding(pos_ids).unsqueeze(0)

    current = init_noise.clone()

    # Physics setup — fields live under ``am.physics`` (PhysicsHead).
    physics_hist_tok = None
    physics_fut = None
    if use_physics:
        physics = am.physics
        if physics_hist is not None and physics.physics_hist_len > 0:
            physics_hist_tok = physics.physics_cond_encoder(physics_hist)
        else:
            physics_hist_tok = torch.zeros(B, 0, msat.inner_dim, dtype=vl.dtype, device=vl.device)
        physics_fut = physics_init_noise.clone() if physics_init_noise is not None else None

    for t in range(denoising_steps):
        timestep_val = t / float(denoising_steps)
        timesteps = torch.tensor([timestep_val], device=vl.device, dtype=vl.dtype).expand(B)

        action_features = am.action_encoder(current, timesteps, embodiment_id)
        action_features = action_features + pos_embs
        sa_embs = torch.cat([state_features, action_features], dim=1)

        physics_embs = None
        if physics_hist_tok is not None and physics_fut is not None:
            physics_fut_tok = am.physics.physics_fut_encoder(physics_fut, timesteps)
            physics_embs = torch.cat([physics_hist_tok, physics_fut_tok], dim=1)

        model_output = msat(
            hidden_states=sa_embs,
            encoder_hidden_states=vl,
            timestep=timesteps,
            physics_embs=physics_embs,
        )

        if isinstance(model_output, dict):
            action_output = model_output["action"]
        else:
            action_output = model_output

        pred = am.action_decoder(action_output, embodiment_id)
        pred_velocity = pred[:, -action_horizon:]
        current = current + dt * pred_velocity

        if physics_fut is not None and isinstance(model_output, dict) and "physics" in model_output:
            physics_hidden_fut = model_output["physics"][:, -am.physics.physics_fut_len :]
            physics_pred_vel = am.physics.physics_decoder(physics_hidden_fut)
            physics_fut = physics_fut + dt * physics_pred_vel

    return current


# Main


def main():
    parser = argparse.ArgumentParser(description="Action Head v1.4 (MSAT) benchmark")
    parser.add_argument(
        "--model-type", type=str, default="rldx_1_pretrain", choices=list(MODEL_REGISTRY.keys())
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument(
        "--n-vl",
        type=int,
        default=None,
        help="VL tokens (default: 64 for v1.4-v, 80 for v1.4-all with memory concat)",
    )
    parser.add_argument("--n-state", type=int, default=1)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--iter", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--compile-mode",
        default="max-autotune",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument("--physics-hist-len", type=int, default=None)
    parser.add_argument("--physics-fut-len", type=int, default=None)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)

    B = 1
    N_state = args.n_state
    action_horizon = args.action_horizon
    N_sa = N_state + action_horizon
    denoising_steps = args.denoising_steps

    # =========================================================================
    # Load action head from checkpoint
    # =========================================================================
    print("=" * 80)
    print("Action Head Benchmark — loading checkpoint")
    print("=" * 80)

    components = load_action_model(
        model_type=args.model_type,
        model_path=args.model_path,
        device=args.device,
    )
    dims = components["dims"]
    use_physics = components.get("use_physics", False)
    # Memory-augmented checkpoints concat 16 cog tokens onto the VL stream.
    # Default heuristic: physics-bearing midtrain ckpts also carry memory.
    mode_str = "v1.4-all (physics)" if use_physics else "v1.4-v"
    N_vl = args.n_vl or (80 if use_physics else 64)

    msat_params = sum(p.numel() for p in components["msat"].parameters()) / 1e6
    enc_dec_params = (
        sum(
            sum(p.numel() for p in m.parameters())
            for m in [
                components["state_encoder"],
                components["action_encoder"],
                components["action_decoder"],
                components["position_embedding"],
            ]
        )
        / 1e6
    )

    physics_dim = components.get("physics_dim", 0)
    physics_hist_len = components.get("physics_hist_len", 0)
    physics_fut_len = components.get("physics_fut_len", 0)
    if args.physics_hist_len is not None:
        physics_hist_len = args.physics_hist_len
        use_physics = True
    if args.physics_fut_len is not None:
        physics_fut_len = args.physics_fut_len
        use_physics = True
    n_physics = physics_hist_len + physics_fut_len if use_physics else 0

    print(f"\n  Mode: {mode_str}")
    print(f"  Params: MSAT={msat_params:.1f}M, enc/dec={enc_dec_params:.1f}M")
    print(f"  Pipeline: state_enc(1x) + {denoising_steps}x[action_enc + MSAT + action_dec + euler]")
    print(f"  Tokens: N_vl={N_vl}, N_sa={N_sa}, +1 time_token -> {N_sa + 1}")
    if use_physics:
        print(
            f"  Physics: dim={physics_dim}, hist={physics_hist_len}, "
            f"fut={physics_fut_len}, n_physics={n_physics}"
        )

    # =========================================================================
    # Test inputs
    # =========================================================================
    torch.manual_seed(args.seed)
    vl_embs = torch.randn((B, N_vl, dims["vl_dim"]), device=device, dtype=dtype) * 0.01
    state = torch.randn((B, N_state, dims["max_state_dim"]), device=device, dtype=dtype) * 0.01
    embodiment_id = torch.zeros(B, dtype=torch.long, device=device)

    torch.manual_seed(args.seed + 1)
    init_noise = torch.randn((B, action_horizon, dims["action_dim"]), device=device, dtype=dtype)

    physics_hist = None
    physics_init_noise = None
    if use_physics and n_physics > 0:
        physics_hist = (
            torch.randn((B, physics_hist_len, physics_dim), device=device, dtype=dtype) * 0.01
        )
        physics_init_noise = torch.randn(
            (B, physics_fut_len, physics_dim), device=device, dtype=dtype
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
        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f"  [!] {label}: NaN/Inf detected")
        print(f"  Warming up ({args.warmup} iters)...")
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        print(f"  Benchmarking ({args.iter} iters)...")
        times = measure_times(fn, args.iter)
        results[label] = times
        outputs[label] = output.detach().clone()

    def make_gs_fn(pipeline):
        def fn():
            with torch.no_grad():
                return pipeline(
                    vl_embs,
                    state,
                    embodiment_id,
                    init_noise=init_noise,
                    physics_hist=physics_hist,
                    physics_init_noise=physics_init_noise,
                )

        return fn

    # =========================================================================
    # Path A: Vanilla (eager)
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Path A: Vanilla (action head, eager)")
    print(f"{'=' * 60}")

    def vanilla_fn():
        with torch.no_grad():
            return vanilla_action_head_forward(
                components,
                vl_embs,
                state,
                embodiment_id,
                init_noise,
                action_horizon,
                denoising_steps,
                physics_hist=physics_hist,
                physics_init_noise=physics_init_noise,
            )

    run_benchmark("A: Vanilla", vanilla_fn)

    # =========================================================================
    # Path B: Torch Inductor on Vanilla (compile MSAT in-place)
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f"Path B: Torch Inductor on Vanilla ({args.compile_mode})")
    print(f"{'=' * 60}")
    orig_msat = components["msat"]
    try:
        torch._dynamo.reset()
        compiled_msat = torch.compile(orig_msat, mode=args.compile_mode)
        components["msat"] = compiled_msat

        print("  Triggering compilation...")
        t0 = _time.time()
        with torch.no_grad():
            vanilla_action_head_forward(
                components,
                vl_embs,
                state,
                embodiment_id,
                init_noise,
                action_horizon,
                denoising_steps,
                physics_hist=physics_hist,
                physics_init_noise=physics_init_noise,
            )
        torch.cuda.synchronize()
        build_times["B: Inductor"] = _time.time() - t0
        print(f"  Compilation: {build_times['B: Inductor']:.1f}s")

        run_benchmark("B: Inductor (vanilla)", vanilla_fn)
    except Exception as e:
        print(f"  [Inductor] Failed: {e}")
        traceback.print_exc()
    finally:
        components["msat"] = orig_msat
        torch._dynamo.reset()

    # =========================================================================
    # Path C: GraphSafe + CUDA Graph
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Path C: GraphSafe + CUDA Graph")
    print(f"{'=' * 60}")
    try:
        print("  Building GraphSafeActionModel...")
        physics_override = args.physics_hist_len is not None or args.physics_fut_len is not None
        if physics_override and n_physics > 0:
            gs_action_model = GraphSafeActionModel(
                action_model=components["action_model"],
                n_vl=N_vl,
                n_sa_pure=N_sa,
                action_horizon=action_horizon,
                action_dim=dims["action_dim"],
                num_inference_timesteps=denoising_steps,
                device=device,
                dtype=dtype,
                physics_cond_encoder=components.get("physics_cond_encoder"),
                physics_fut_encoder=components.get("physics_fut_encoder"),
                physics_decoder=components.get("physics_decoder"),
                physics_hist_len=physics_hist_len,
                physics_fut_len=physics_fut_len,
                physics_dim=physics_dim,
            ).eval()
        else:
            gs_action_model = GraphSafeActionModel(
                action_model=components["action_model"],
                n_vl=N_vl,
                n_sa_pure=N_sa,
                action_horizon=action_horizon,
                action_dim=dims["action_dim"],
                num_inference_timesteps=denoising_steps,
                device=device,
                dtype=dtype,
            ).eval()

        cuda_graph_msat = setup_cuda_graph(gs_action_model.gs_msat)
        orig_gs_msat = gs_action_model.gs_msat
        gs_action_model.gs_msat = cuda_graph_msat

        print("  Capturing CUDA graph (warmup + capture)...")
        with torch.no_grad():
            gs_action_model(
                vl_embs,
                state,
                embodiment_id,
                physics_hist=physics_hist,
                physics_init_noise=physics_init_noise,
            )
        torch.cuda.synchronize()
        print("  CUDA graph captured")

        run_benchmark("C: GraphSafe + CG", make_gs_fn(gs_action_model))
        gs_action_model.gs_msat = orig_gs_msat
    except Exception as e:
        print(f"  [GraphSafe + CG] Failed: {e}")
        traceback.print_exc()

    # =========================================================================
    # Path D: CustomActionHeadChain + torch.compile
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Path D: CustomActionHeadChain + torch.compile")
    print(f"{'=' * 60}")
    try:
        if "gs_action_model" not in locals():
            gs_action_model = GraphSafeActionModel(
                action_model=components["action_model"],
                n_vl=N_vl,
                n_sa_pure=N_sa,
                action_horizon=action_horizon,
                action_dim=dims["action_dim"],
                num_inference_timesteps=denoising_steps,
                device=device,
                dtype=dtype,
            ).eval()

        if n_physics > 0:
            from engine.custom_expanded_action_model_chain import CustomExpandedActionHeadChain

            print("  Building CustomExpandedActionHeadChain (3-way)...")
            custom_ah = CustomExpandedActionHeadChain(
                gs_action_model, device=device, dtype=dtype
            ).eval()
        else:
            from engine.custom_action_model_chain import CustomActionHeadChain

            print("  Building CustomActionHeadChain (2-way)...")
            custom_ah = CustomActionHeadChain(gs_action_model, device=device, dtype=dtype).eval()

        print(f"  Compiling (mode={args.compile_mode})...")
        compiled_ah = torch.compile(custom_ah, mode=args.compile_mode)

        t0 = _time.time()
        with torch.no_grad():
            for i in range(5):
                if n_physics > 0:
                    compiled_ah(
                        vl_embs,
                        state,
                        embodiment_id,
                        init_noise=init_noise,
                        physics_hist=physics_hist,
                        physics_init_noise=physics_init_noise,
                    )
                else:
                    compiled_ah(vl_embs, state, embodiment_id, init_noise=init_noise)
                if i == 0:
                    print("    Compilation warmup 1/5 done")
        torch.cuda.synchronize()
        build_times["D: CustomChain"] = _time.time() - t0
        print(f"    Compilation complete ({build_times['D: CustomChain']:.1f}s)")

        run_benchmark("D: CustomActionHeadChain", make_gs_fn(compiled_ah))
    except Exception as e:
        print(f"  [CustomChain] Failed: {e}")
        traceback.print_exc()

    # =========================================================================
    # Report
    # =========================================================================
    print()
    print("=" * 80)
    print(f"Action Head Benchmark — {mode_str}")
    print("=" * 80)
    print(f"Config: H=24, D=64, inner_dim={dims['sa_dim']}, output_dim={dims['hidden_size']}")
    if use_physics:
        print(
            f"Physics: dim={physics_dim}, hist={physics_hist_len}, "
            f"fut={physics_fut_len}, n_physics={n_physics}"
        )
    print(f"Pipeline: state_enc(1x) + {denoising_steps}x[action_enc + MSAT + action_dec + euler]")
    print(f"Tokens: N_vl={N_vl}, N_sa={N_sa}+1time")
    print(f"Params: MSAT={msat_params:.1f}M, enc/dec={enc_dec_params:.1f}M")
    print(f"Iter={args.iter}, warmup={args.warmup}")
    if build_times:
        print(f"Build: {', '.join(f'{k}={v:.1f}s' for k, v in build_times.items())}")

    print_latency_table(
        f"Latency ({denoising_steps} denoising steps)",
        results,
    )

    if "A: Vanilla" in outputs and len(outputs) > 1:
        vanilla_out = outputs["A: Vanilla"]
        corr_entries = [
            (label, out, None) for label, out in outputs.items() if label != "A: Vanilla"
        ]
        print_correctness("Correctness (vs A: Vanilla)", corr_entries, vanilla_out)

    print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated(device) / (1024**2):.1f} MB\n")


if __name__ == "__main__":
    main()
