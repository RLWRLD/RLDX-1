# Backbone Backbone Benchmark

Full Backbone pipeline benchmark (Vision Encoder + RoPE + LLM Decoder + MetaQuery projection).

B=1, 2x224 images, seq_len=169, hidden=4096, 18 LLM layers, RTX 5090.

## Benchmark Results

### v1.4-v

| Path | p50 (ms) | mean (ms) | std (ms) | speedup |
|------|:--------:|:---------:|:--------:|:-------:|
| A: Vanilla | 140.83 | 161.54 | 65.89 | — |
| B: Torch Inductor (vanilla) | 125.66 | 151.87 | 69.37 | 1.12x |
| C: GraphSafe + CUDA Graph | 18.37 | 18.38 | 0.02 | 7.67x |
| **D: CustomVLMChain** | **15.23** | **15.24** | **0.09** | **9.25x** |

Correctness (vs A: Vanilla):

| Path | max_diff | cos_sim | allclose |
|------|:--------:|:-------:|:--------:|
| B: Inductor | 0.125000 | 0.99998277 | False |
| C: GraphSafe + CG | 0.000000 | 1.00000000 | True |
| D: CustomVLMChain | 0.250000 | 0.99997061 | False |

### v1.4-all

| Path | p50 (ms) | mean (ms) | std (ms) | speedup |
|------|:--------:|:---------:|:--------:|:-------:|
| A: Vanilla | 139.21 | 157.11 | 64.64 | — |
| B: Torch Inductor (vanilla) | 129.74 | 148.51 | 63.12 | 1.07x |
| C: GraphSafe + CUDA Graph | 18.79 | 18.80 | 0.07 | 7.41x |
| **D: CustomVLMChain** | **15.32** | **15.32** | **0.14** | **9.09x** |

Correctness (vs A: Vanilla):

| Path | max_diff | cos_sim | allclose |
|------|:--------:|:-------:|:--------:|
| B: Inductor | 0.250000 | 0.99999470 | False |
| C: GraphSafe + CG | 0.000000 | 0.99999994 | True |
| D: CustomVLMChain | 0.328125 | 0.99999404 | False |

## Optimization Paths

| Path | Technique | Description |
|------|-----------|-------------|
| A: Vanilla | None | Original PyTorch model, eager execution (baseline) |
| B: Torch Inductor | Compiler only | `torch.compile` on vanilla LLM layers |
| C: GraphSafe + CUDA Graph | Static wrapping + CUDA Graph | Wrap model as static graph, then apply CUDA Graph capture |
| D: CustomVLMChain | Path C + Kernel Fusion | Path C base + custom optimized Triton fused kernels |

## Usage

```bash
# v1.4-v (default)
python inference/backbone/benchmark_backbone.py

# v1.4-all
python inference/backbone/benchmark_backbone.py --mode all

# Custom options
python inference/backbone/benchmark_backbone.py --mode video --iter 300 --warmup 100 --device 0
```
