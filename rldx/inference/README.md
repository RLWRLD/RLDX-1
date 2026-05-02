# RLDX Inference Optimization

Optimized inference library for RLDX v1.4 models.
Preserves mathematically identical results while improving GPU execution speed.

## Getting Started

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| NVIDIA Driver | >= 580.82.07 | Must support CUDA 13.0 |
| CUDA Toolkit (NVCC) | 13.0 | `nvcc --version` to verify |
| Python | 3.10 | |
| GPU | RTX 5090 (sm_120 / Blackwell) | Single GPU, 32 GB |

> **Why CUDA 13.0?** Optimization paths B/D use Torch Inductor (`torch.compile`)
> to generate PTX/CUBIN targeting the GPU's native architecture.
> If the NVCC version does not support the target architecture (sm_120 for Blackwell),
> the compile pass will fail. CUDA 13.0 is the minimum toolkit version that includes
> sm_120 support.

> **PyTorch version:** The upstream GR00T repository runs on **PyTorch 2.7.0**.
> This inference optimization suite requires **PyTorch 2.10.0+cu130** — the two
> environments are not interchangeable. Use the setup script below to install
> the correct build.

### Environment Setup

```bash
# 1. Set CUDA 13.0 paths
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH

# 2. Install dependencies (torch 2.10.0+cu130, flash-attn source build)
bash scripts/setup_inference_env.sh

# 3. Verify
.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)"
# Expected: 2.10.0+cu130 13.0
```

`scripts/setup_inference_env.sh` runs `uv sync` with the project's `pyproject.toml`, which pins
`torch==2.10.0` and `torchvision==0.25.0` from the PyTorch cu130 wheel index,
and builds `flash-attn` from source with `FLASH_ATTN_CUDA_ARCHS="120"`.

## Run Benchmarks

```bash
# v1.4-v
python inference/benchmark_vla.py --model-type rldx_1_pretrain --num-images 2

# v1.4-all
python inference/benchmark_vla.py --model-type rldx_1_midtrain_allex --num-images 2
```

Each benchmark script runs four optimization paths in order:

| Path | Technique | Description |
|------|-----------|-------------|
| **A: Vanilla** | None | Original PyTorch model, eager execution (baseline) |
| **B: Torch Inductor** | Compiler only | `torch.compile` on the vanilla model's sub-modules |
| **C: GraphSafe + CUDA Graph** | Static wrapping + CUDA Graph | Wrap model as static graph, then apply CUDA Graph capture |
| **D: Custom Chain** | Path C + Kernel Fusion | Path C base + custom optimized Triton fused kernels |

> **Note on TensorRT:** The original model's dynamic control flow prevents direct
> TensorRT conversion. TensorRT is excluded from the current benchmark suite.

For detailed per-module benchmarks, see:
- `backbone/benchmark_backbone.py` — Backbone backbone (Vision Encoder + LLM Decoder)
- `action_model/benchmark_action_model.py` — Action Model (MSAT)

## Benchmark Environment

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GeForce RTX 5090 (32 GB, sm_120 / Blackwell) |
| CPU | 2× Intel Xeon E5-2620 v4 @ 2.10 GHz (16 cores / 32 threads) |
| RAM | 188 GiB |
| OS | Ubuntu 22.04.5 LTS (kernel 5.15.0-168-generic) |
| CUDA | 13.0 |
| PyTorch | 2.10.0+cu130 |
| torchvision | 0.25.0+cu130 |
| Triton | 3.6.0 |
| Flash Attention | 2.7.4.post1 |

## Benchmark Results

4 denoising steps, B=1, RTX 5090.

### VLA Full Pipeline (Backbone + Memory + Action Model)

| Path | v1.4-v p50 (ms) | v1.4-all p50 (ms) |
|------|:---------------:|:-----------------:|
| A: Vanilla | 191.19 | 193.23 |
| B: Torch Inductor (vanilla) | 189.48 (1.01x) | 189.07 (1.02x) |
| C: GraphSafe + CUDA Graph | 33.70 (5.67x) | 36.34 (5.32x) |
| **D: Custom Chain** | **25.20 (7.59x)** | **26.81 (7.21x)** |

Correctness (cos_sim vs A: Vanilla):

| Path | v1.4-v | v1.4-all |
|------|:------:|:--------:|
| B: Torch Inductor | 0.99997 | 0.99999 |
| C: GraphSafe + CG | 1.00000 | 0.99997 |
| D: Custom Chain | 0.99997 | 0.99997 |

## Apply Optimization

### 1. Load Model

```python
from utils import load_vla

backbone, action_head, meta = load_vla(
    model_type="rldx_1_midtrain_allex",  # or "rldx_1_pretrain"
    device=0,
)
```

`load_vla` extracts backbone, action_head, and (optional) memory from a HuggingFace checkpoint.
`meta` contains `memory_module`, `memory_config`, `use_physics`, etc.

### 2. Build GraphSafe Model

```python
from model.graph_safe_vla import GraphSafeVLA
from vlm.model import GraphSafeQwen3VLBackbone
from action_head.model import GraphSafeActionModel
from memory.model import GraphSafeMemory

# Backbone
gs_backbone = GraphSafeQwen3VLBackbone(backbone, vl_input)

# Memory (optional, v1.4-all)
gs_memory = None
if meta.get("memory_module") is not None:
    gs_memory = GraphSafeMemory(
        memory_module=meta["memory_module"],
        memory_length=meta["memory_config"]["memory_length"],
        memory_n_meta_queries=meta["memory_config"]["memory_n_meta_queries"],
        device=device, dtype=torch.bfloat16,
    )

# Action Model
gs_action_model = GraphSafeActionModel(
    action_head=action_head,
    n_vl=n_vl, n_sa_pure=N_sa,
    action_horizon=16, action_dim=64,
    num_inference_timesteps=4,
    device=device, dtype=torch.bfloat16,
)

# VLA (unified)
gs_vla = GraphSafeVLA(gs_backbone, gs_action_model,
                       gs_memory=gs_memory,
                       memory_config=meta.get("memory_config"))
```

### 3. Apply Optimization Path

#### Path C: GraphSafe + CUDA Graph

Wraps the model as a static graph (GraphSafe), then captures the full VLA pipeline
as a CUDA Graph to eliminate kernel launch overhead.

```python
from engine.cudaGraph import setup_vla_cuda_graph

replay_fn, _ = setup_vla_cuda_graph(gs_vla, vl_input, state, embodiment_id)

with torch.no_grad():
    action = replay_fn(vl_input, state, embodiment_id, init_noise=init_noise)
```

#### Path D: Custom Chain (Path C + Kernel Fusion)

Builds on Path C by adding custom optimized Triton fused kernels per module
(Backbone, Action Model), then compiles the entire chain as a single unit via
`torch.compile` (Torch Inductor).

```python
from engine.custom_vla_chain import build_custom_vla_chain, compile_custom_vla_chain

# Build (auto-selects v1.4-v / v1.4-all)
vla_chain = build_custom_vla_chain(gs_vla, device, dtype=torch.bfloat16)

# Compile
sample_inputs = (pixel_values, state, embodiment_id, init_noise)
compiled_vla, compile_time = compile_custom_vla_chain(
    vla_chain, sample_inputs, compile_mode="max-autotune")

with torch.no_grad():
    action = compiled_vla(pixel_values, state, embodiment_id, init_noise=init_noise)
```

`build_custom_vla_chain` auto-selects based on model type:
- v1.4-v: 2-way chains (DoubleStreamBlock + SingleStreamBlock)
- v1.4-all: 3-way chains (ExpandedDoubleStreamBlock + ExpandedSingleStreamBlock)

When memory is present, the pipeline is Backbone → Memory → ActionHead.

## Optimization Techniques

### 1. GraphSafe Model Wrapping

The original VLA model cannot be expressed as a static computation graph due to
data-dependent control flow (dynamic shapes, conditional branches, in-place ops, etc.).
This prevents CUDA Graph capture and limits compiler optimization.

GraphSafe models wrap the original modules, pre-computing data-dependent values
(position IDs, attention masks, RoPE embeddings) as static buffers at construction time.
The computation is mathematically identical, but the forward pass becomes a pure
static graph — enabling CUDA Graph capture and full compiler optimization.

### 2. Custom Kernel Fusion

Hand-written Triton kernels fuse multiple operations (norm + RoPE + QKV projection,
grouped attention epilogues, SwiGLU FFN, etc.) into single GPU kernel launches,
eliminating memory round-trips between operators. These fused kernels are assembled
into module-level chains and compiled as a single unit via `torch.compile` (Torch Inductor),
forming a unified computation engine per module.

## Directory Structure

```
inference/
├── _path.py                          Path bootstrap for scripts
├── benchmark_vla.py                  Full VLA pipeline benchmark
│
├── model/
│   └── graph_safe_vla.py             GraphSafeVLA
│
├── engine/
│   ├── cudaGraph.py                  CUDA Graph
│   ├── torchInductor.py              torch.compile
│   └── custom_vla_chain.py           Custom VLA Chain
│
├── backbone/
│   ├── benchmark_vlm.py
│   ├── model/                        GraphSafeQwen3VLBackbone
│   ├── engine/                       Backbone optimization engines
│   ├── vision_encoder/engine/        Vision encoder kernels
│   └── llm/engine/                   LLM decoder kernels
│
├── action_model/
│   ├── benchmark_action_head.py
│   ├── model/                        GraphSafeActionModel, GraphSafeMSAT
│   ├── engine/                       ActionHead + MSAT chains
│   ├── double_stream/engine/         DoubleStream kernels
│   └── single_stream/engine/         SingleStream kernels
│
└── utils/
    ├── loader.py                     load_vla, load_vlm_backbone, load_action_head, load_memory
    ├── registry.py                   Model registry (checkpoint paths)
    ├── input_generator.py            Synthetic input generation
    ├── timing.py                     Latency measurement
    └── correctness.py                Correctness comparison
```

## Supported Models

| Model | Checkpoint | Memory | Physics |
|-------|-----------|:------:|:-------:|
| v1.4-v | `kimtaey/alinvla_v1d4_video_mq64_pt100k_ft_robocasa_bsz512_60k` | - | - |
| v1.4-all | `kimtaey/g0_midtrained_allex_50k` | O | structure only (n_physics=0) |
