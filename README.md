<div align="center">

# RLDX-1

[[Paper]](<PAPER_TBD>) [[Project Page]](https://www.rlwrld.ai/rldx-1) [[Models]](https://huggingface.co/collections/RLWRLD/rldx-1)

<img src="assets/teaser.png" width="100%" alt="RLDX-1 teaser">

</div>

---

RLDX-1 is a Vision-Language-Action model (VLA) for human-like
dexterous manipulation. Beyond the *versatile intelligence* inherited
from pre-trained VLM backbones, RLDX-1 adds three **functional
capabilities** — motion awareness, long-term memory, and physical
sensing — through a unified **Multi-Stream Action Transformer (MSAT)**
architecture, a synthetic-augmented training pipeline, and a real-time
inference stack.

---

<div align="center">
<img src="assets/architecture.png" width="90%" alt="RLDX-1 architecture">
</div>

## Highlights

- **Multi-Stream Action Transformer (MSAT).** Cognition, physics, and
  action each get a dedicated stream coupled by joint self-attention —
  an extension of MM-DiT to action modeling.
- **Motion awareness.** Multi-frame observations + a motion module
  capture temporal dynamics; intermediate VLM layers compress video
  tokens to keep the policy efficient.
- **Long-term memory.** A memory module fuses past cognition features
  with the current ones for history-grounded decisions beyond a short
  multi-frame window.
- **Physical sensing.** Tactile and torque enter as a dedicated physics
  stream; the decoder is jointly trained to predict future physical
  signals.
- **Three-stage training.** Pre-training (generalization) → mid-training
  (functionality) → post-training (task adaptation), with synthetic data
  augmenting rare manipulation scenarios.
- **Real-time inference.** Static graph capture + custom fused kernels
  bring the all-modality model to **43.7 ms / step on RTX 5090
  (1.63× speedup, >22 Hz)**.

---

## Performance

### Simulation Benchmarks

Success rates (%) of RLDX-1 fine-tuned on each benchmark's training set,
compared to recent frontier VLA baselines. Numbers from the
[RLDX-1 Technical Report](<PAPER_TBD>) (Table 1).

| Method | LIBERO (Avg) | LIBERO-Plus | SIMPLER Google-VM | SIMPLER Google-VA | SIMPLER WidowX | RoboCasa Kitchen | GR-1 Tabletop | RoboCasa365 (Avg) |
|---|---|---|---|---|---|---|---|---|
| π0-FAST | 85.5 | 64.2 | 61.9 | 59.0 | 48.3 | 63.6 | — | 21.7 |
| π0      | 94.1 | 54.6 | 58.8 | 54.8 | 27.1 | 62.5 | 13.6 | 14.8 |
| π0.5    | 96.9 | 86.5 | 72.7 | 68.4 | 46.9 | 62.1 | 15.4 | 16.9 |
| GR00T N1.5 | 86.5 | 66.3 | 52.4 | 43.7 | 62.0 | 65.7 | 48.0 | 20.0 |
| GR00T N1.6 | 96.7 | 72.6 | 76.1 | 57.1 | 57.1 | 66.2 | 47.6 | 26.9 |
| **RLDX-1 (ours)** | **97.8** | **86.7** | **81.5** | **77.4** | **71.9** | **70.6** | **58.7** | **32.1** |

The first five columns cover the established LIBERO / SIMPLER family;
the last three (RoboCasa Kitchen, GR-1 Tabletop, RoboCasa365) are
long-horizon, humanoid, and compositional benchmarks. Per-benchmark
checkpoints, embodiment tags, and reproduce commands are listed under
[Reproducing Benchmark Results](#reproducing-benchmark-results).

---

## Installation

**Requirements**: Python 3.10, CUDA 12.x, [uv](https://github.com/astral-sh/uv) v0.8.4+

```bash
git clone https://github.com/RLWRLD/RLDX.git
cd RLDX
uv sync --python 3.10
uv pip install -e .
```

Verify installation:
```bash
uv run python -c "import rldx; print(rldx.__version__)"
```

For simulator setup, dev tooling, and full troubleshooting, see
[`docs/installation.md`](docs/installation.md).

---

## Documentation

Hands-on guides live under [`docs/`](docs/):

| Guide | What it covers |
|---|---|
| [`installation.md`](docs/installation.md) | Environment setup, simulator venvs, dev tooling, common pitfalls |
| [`architecture.md`](docs/architecture.md) | Five-stage walkthrough of the RLDX-1 model and its config flags |
| [`training.md`](docs/training.md) | `launch_train.py` recipes (finetune / pre-train / mid-train), LoRA, training-time RTC, dataset layout |
| [`embodiment_tags.md`](docs/embodiment_tags.md) | What `EmbodimentTag` is and how to pick one for a custom robot |
| [`evaluation.md`](docs/evaluation.md) | RoboCasa / LIBERO / SIMPLER / GR-1 eval, server + rollout split, results aggregation |
| [`inference_server.md`](docs/inference_server.md) | `run_rldx_server.py` CLI, wire protocol, RTC modes, `--compile` levels, simulator + real-robot deployment |


---

## Pretrained & Midtrained Checkpoints

| Checkpoint | Description | Params | HuggingFace |
|-----------|-------------|--------|-------------|
| `RLDX-1-PT` | Pre-trained (video) | 6.9B | [RLWRLD/RLDX-1-PT](https://huggingface.co/RLWRLD/RLDX-1-PT) |
| `RLDX-1-MT-DROID` | Mid-trained on DROID with motion + memory + physical-sensing add-ons | 8.1B | [RLWRLD/RLDX-1-MT-DROID](https://huggingface.co/RLWRLD/RLDX-1-MT-DROID) |
| `RLDX-1-MT-ALLEX` | Mid-trained on ALLEX with all add-ons | 8.1B | [RLWRLD/RLDX-1-MT-ALLEX](https://huggingface.co/RLWRLD/RLDX-1-MT-ALLEX) |

---

## Data Preparation

RLDX uses [LeRobot](https://github.com/huggingface/lerobot) v2.1 format datasets. To convert your data:

```bash
# Convert a single dataset
bash run_scripts/data/convert_lerobot_single.sh /path/to/your/data

# Convert multiple datasets
bash run_scripts/data/convert_lerobot_multiple.sh /path/to/data/root
```

Each dataset must carry a `meta/modality.json` that slices the flat
state / action vectors into named joint groups and remaps video columns
to modality keys. Schema and a worked example are in
[`docs/training.md`](docs/training.md#dataset-layout-metamodalityjson).

### Custom Embodiment Config

Define your robot's modality configuration:

```python
# my_modality_config.py
from rldx.data.types import ModalityConfig

MODALITY_CONFIGS = {
    "my_robot": {
        "image": ModalityConfig(...),
        "state": ModalityConfig(...),
        "action": ModalityConfig(...),
    }
}
```

Pass it via `--modality-config-path my_modality_config.py` during training,
together with an `EmbodimentTag` that selects the per-robot MLP head slot
(default: `GENERAL_EMBODIMENT`; see
[`docs/embodiment_tags.md`](docs/embodiment_tags.md) for the picker).

---

## Finetuning

### Single Dataset

```bash
uv run python rldx/experiment/launch_train.py \
    --base-model-path RLWRLD/RLDX-1-PT \
    --dataset-path /path/to/your/dataset \
    --embodiment-tag GENERAL_EMBODIMENT \
    --video-length 4 \
    --n-cog-tokens 64 \
    --global-batch-size 64 \
    --learning-rate 1e-4 \
    --max-steps 60000 \
    --save-steps 5000 \
    --output-dir ./outputs/my_finetune
```

### With Memory Module

```bash
uv run python rldx/experiment/launch_train.py \
    --base-model-path RLWRLD/RLDX-1-PT \
    --dataset-path /path/to/your/dataset \
    --embodiment-tag GENERAL_EMBODIMENT \
    --video-length 4 \
    --use-memory --memory-length 4 \
    --n-cog-tokens 64 \
    --global-batch-size 64 \
    --max-steps 60000 \
    --output-dir ./outputs/my_finetune_memory
```

### With All Functional Add-ons (Memory + Motion + Physics)

```bash
uv run python rldx/experiment/launch_train.py \
    --base-model-path RLWRLD/RLDX-1-PT \
    --dataset-path /path/to/your/dataset \
    --embodiment-tag GENERAL_EMBODIMENT \
    --video-length 4 \
    --use-memory --memory-length 4 --concat-memory \
    --use-motion --motion-insert-layer 9 \
    --use-physics --physics-keys tactile torque --physics-dims 30 7 \
    --new-param-warmup-steps 2000 \
    --n-cog-tokens 64 \
    --global-batch-size 64 \
    --max-steps 60000 \
    --output-dir ./outputs/my_finetune_all
```

### Key Training Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--video-length` | Number of video frames (VTC backbone is always on; set to `1` for single-frame) | `4` |
| `--video-stride` | Stride between frames in action-step units | `2` |
| `--use-memory` | Enable temporal memory module | `False` |
| `--memory-length` | Memory context window (timesteps) | `4` |
| `--use-motion` | Enable motion module | `False` |
| `--use-physics` | Enable physics signal conditioning | `False` |
| `--n-cog-tokens` | Number of cognition tokens | `64` |
| `--global-batch-size` | Total batch size across GPUs | `64` |
| `--new-param-warmup-steps` | Warmup steps for newly added modules | `0` |

### LoRA finetuning

For memory-constrained finetunes you can replace full-parameter tuning
of the action model (MSAT) and/or the backbone (Qwen3 LLM) with PEFT
LoRA adapters:

```bash
--action-model-use-lora --action-model-lora-rank 16 --action-model-lora-alpha 32
--backbone-use-lora --backbone-lora-rank 16 --backbone-lora-alpha 32 --backbone-lora-num-layers -1
```

`--action-model-use-lora` overrides `--tune-diffusion-model`;
`--backbone-use-lora` overrides `--tune-top-llm-layers`. Full flag list
and target-module defaults are in
[`docs/training.md`](docs/training.md#lora-finetuning).

### Training-time Real-Time Chunking

If you intend to serve the checkpoint with `--rtc-inference-mode trained`
(faster, fullgraph-compatible), enable training-time RTC at training
time:

```bash
--rtc-training-max-delay 4
```

See [`docs/training.md`](docs/training.md#real-time-chunking-training-time)
and [`docs/inference_server.md`](docs/inference_server.md#real-time-chunking-rtc)
for the inference-side counterpart.

---

## Inference

### Quick Start

```python
import torch
from rldx.policy.rldx_policy import RLDXPolicy
from rldx.data.embodiment_tags import EmbodimentTag

policy = RLDXPolicy(
    model_path="RLWRLD/RLDX-1-FT-ROBOCASA",
    embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT,
    device="cuda:0",
)

# Single-step inference
action = policy.get_action(observation)
```

### Serving (ZeroMQ)

For real-time robot deployment:

```bash
# Start the policy server
uv run python rldx/eval/run_rldx_server.py \
    --model-path RLWRLD/RLDX-1-FT-ROBOCASA \
    --embodiment-tag GENERAL_EMBODIMENT \
    --host 0.0.0.0 --port 20000
```

### Real-time inference (graph capture + RTC)

The server brings the all-modality model to **43.7 ms / step on RTX
5090 (1.63× speedup, >22 Hz)** through two orthogonal knobs:

- `--compile {none, submodule, fullgraph}` — `submodule` compiles each
  learnable sub-module (preserves autograd, ~30 s warmup); `fullgraph`
  applies CUDA-graph capture and operator fusion across the full VLA
  forward (~90–210 s warmup, lowest steady-state latency).
- `--rtc-inference-mode {none, guided, trained}` — Real-Time Chunking
  for chunk-boundary stitching. `guided` works with any flow-matching
  checkpoint; `trained` requires a checkpoint trained with
  `--rtc-training-max-delay > 0` and pairs with `--compile fullgraph`.

The full flag list, the `compile × RTC` compatibility matrix, and a
walkthrough of the trade-offs are in
[`docs/inference_server.md`](docs/inference_server.md#real-time-chunking-rtc).

---

## Reproducing Benchmark Results

Each benchmark has a self-contained eval README; this table maps each
result row in [Performance](#simulation-benchmarks) to the fine-tuned
checkpoint we used, the embodiment tag the server expects, and the
runnable guide.

| Benchmark | Fine-tuned Checkpoint | Embodiment Tag | Eval Guide |
|---|---|---|---|
| LIBERO | [RLWRLD/RLDX-1-FT-LIBERO](https://huggingface.co/RLWRLD/RLDX-1-FT-LIBERO) | `GENERAL_EMBODIMENT` | [`run_scripts/eval/libero/README.md`](run_scripts/eval/libero/README.md) |
| LIBERO-Plus | [RLWRLD/RLDX-1-FT-LIBERO](https://huggingface.co/RLWRLD/RLDX-1-FT-LIBERO) | `GENERAL_EMBODIMENT` | [`run_scripts/eval/libero_plus/README.md`](run_scripts/eval/libero_plus/README.md) |
| SimplerEnv Google | [RLWRLD/RLDX-1-FT-SIMPLER-GOOGLE](https://huggingface.co/RLWRLD/RLDX-1-FT-SIMPLER-GOOGLE) | `OXE_FRACTAL` | [`run_scripts/eval/simpler/README.md`](run_scripts/eval/simpler/README.md) |
| SimplerEnv WidowX | [RLWRLD/RLDX-1-FT-SIMPLER-WIDOWX](https://huggingface.co/RLWRLD/RLDX-1-FT-SIMPLER-WIDOWX) | `OXE_BRIDGE_ORIG` | [`run_scripts/eval/simpler/README.md`](run_scripts/eval/simpler/README.md) |
| GR-1 Tabletop | [RLWRLD/RLDX-1-FT-GR1](https://huggingface.co/RLWRLD/RLDX-1-FT-GR1) | `GENERAL_EMBODIMENT` | [`run_scripts/eval/gr1_tabletop/README.md`](run_scripts/eval/gr1_tabletop/README.md) |
| RoboCasa Kitchen (24 tasks) | [RLWRLD/RLDX-1-FT-ROBOCASA](https://huggingface.co/RLWRLD/RLDX-1-FT-ROBOCASA) | `GENERAL_EMBODIMENT` | [`run_scripts/eval/robocasa_kitchen/README.md`](run_scripts/eval/robocasa_kitchen/README.md) |
| RoboCasa365 | [RLWRLD/RLDX-1-FT-RC365](https://huggingface.co/RLWRLD/RLDX-1-FT-RC365) | `GENERAL_EMBODIMENT` | [`run_scripts/eval/robocasa_365/README.md`](run_scripts/eval/robocasa_365/README.md) |

Shared mechanics (server + rollout split, common flags, troubleshooting)
are documented in [`docs/evaluation.md`](docs/evaluation.md).

---

## Project Structure

```
rldx/
├── configs/                              # Model, data, and training configurations
├── data/                                 # Dataset loaders, processors, and statistics
├── experiment/                           # Training entry points and utilities
├── eval/                                 # Evaluation scripts and sim environments
├── model/
│   ├── rldx/                             # Core model (RLDX, processor, pipeline)
│   └── modules/
│       ├── backbone/                     # Vision-language backbones
│       ├── action_model/                 # MSAT diffusion action model
│       ├── vtc/                          # VTC video backbone
│       ├── common.py                     # Shared primitives (RMSNorm, etc.)
│       ├── memory.py                     # Temporal memory transformer
│       ├── physical_signal.py            # Physics encoder/decoder
│       └── embodiment_conditioned_mlp.py
├── policy/                               # Inference policy wrappers
└── utils/                                # Distributed training utilities
```

---

## Citation

```bibtex
@article{rldx2026,
  title={RLDX-1 Technical Report},
  author={Koo, Myungkyu and Jang, Suhyeok and Kim, Taeyoung and others},
  year={2026},
  note={RLWRLD},
  url={https://github.com/RLWRLD/RLDX}
}
```

---

## Acknowledgments

RLDX builds upon the following open-source projects:

- [NVIDIA GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.7-release) — Training Codebase
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) — Vision-language backbone
- [ContextVLA](https://arxiv.org/abs/2510.04246) — Multi-frame video compression
- [Flux](https://github.com/black-forest-labs/flux) — MMDiT architecture
- [HAMLET](https://arxiv.org/abs/2510.00695) — Memory-augmented MetaQueries

## License

RLDX is released under the [Apache License 2.0](LICENSE). The model
derives from NVIDIA Isaac GR00T v1.7 (also Apache 2.0); third-party
attributions and per-file provenance headers are preserved in the
source tree.
