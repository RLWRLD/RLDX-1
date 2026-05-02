# Action Model (MSAT) Benchmark

Full action head pipeline benchmark (state_enc + 4x[action_enc + MSAT + action_dec + euler]).

B=1, 4 denoising steps, RTX 5090.

## Benchmark Results

### v1.4-v

Config: H=24, D=64, inner_dim=1536, N_vl=64, N_sa=17+1time, MSAT=1244.6M params.

| Path | p50 (ms) | mean (ms) | std (ms) | speedup |
|------|:--------:|:---------:|:--------:|:-------:|
| A: Vanilla | 149.27 | 166.97 | 59.86 | — |
| B: Torch Inductor (vanilla) | 17.43 | 20.25 | 5.99 | 8.56x |
| C: GraphSafe + CUDA Graph | 15.42 | 15.59 | 1.09 | 9.68x |
| **D: CustomActionHeadChain** | **9.66** | **9.66** | **0.02** | **15.45x** |

Correctness (vs A: Vanilla):

| Path | max_diff | cos_sim | allclose |
|------|:--------:|:-------:|:--------:|
| B: Inductor | 0.015625 | 0.99999648 | False |
| C: GraphSafe + CG | 0.000000 | 1.00000000 | True |
| D: CustomActionHeadChain | 0.015625 | 0.99999654 | False |

### v1.4-all

Config: H=24, D=64, inner_dim=1536, N_vl=80, N_sa=17+1time, MSAT=1842.6M params (ExpandedStreamBlocks).

| Path | p50 (ms) | mean (ms) | std (ms) | speedup |
|------|:--------:|:---------:|:--------:|:-------:|
| A: Vanilla | 146.53 | 154.12 | 52.01 | — |
| B: Torch Inductor (vanilla) | 19.11 | 21.24 | 6.85 | 7.67x |
| C: GraphSafe + CUDA Graph | 16.82 | 18.36 | 4.37 | 8.71x |
| **D: CustomActionHeadChain** | **10.25** | **10.26** | **0.19** | **14.30x** |

Correctness (vs A: Vanilla):

| Path | max_diff | cos_sim | allclose |
|------|:--------:|:-------:|:--------:|
| B: Inductor | 0.068359 | 0.99977934 | False |
| C: GraphSafe + CG | 0.000000 | 0.99999988 | True |
| D: CustomActionHeadChain | 0.064453 | 0.99979246 | False |

## Optimization Paths

| Path | Technique | Description |
|------|-----------|-------------|
| A: Vanilla | None | Original PyTorch model, eager execution (baseline) |
| B: Torch Inductor | Compiler only | `torch.compile` on vanilla MSAT |
| C: GraphSafe + CUDA Graph | Static wrapping + CUDA Graph | Wrap model as static graph, then apply CUDA Graph capture |
| D: CustomActionHeadChain | Path C + Kernel Fusion | Path C base + custom optimized Triton fused kernels |

## Usage

```bash
# v1.4-v (default)
python inference/action_model/benchmark_action_model.py

# v1.4-all
python inference/action_model/benchmark_action_model.py --model-type rldx_1_midtrain_allex

# Custom options
python inference/action_model/benchmark_action_model.py --model-type rldx_1_pretrain --iter 300 --warmup 100 --device 0
```
