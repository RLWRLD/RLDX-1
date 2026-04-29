<div align="center">

# RLDX-1

[[Paper]](<PAPER_TBD>) [[Project Page]](https://www.rlwrld.ai/rldx-1) [[Models]](https://huggingface.co/collections/RLWRLD/rldx-1)

<img src="assets/teaser.png" width="100%" alt="RLDX-1 teaser">

</div>

---

RLDX-1 is a general-purpose Robot Foundation Model designed for dexterous manipulation. Powered by a **Multi-Stream Action Transformer (MSAT)**, it seamlessly unifies multimodal perception (visual + tactile), high-DoF actuation, and memory-aware decision-making into a single architecture. RLDX-1 achieves state-of-the-art performance across diverse simulation benchmarks and is fully validated on real-world hardware.

---

<div align="center">
<img src="assets/architecture.png" width="90%" alt="RLDX-1 architecture">
</div>

## Highlights

- **Unified MSAT policy.** Vision, language, proprioception, and physical
  signals fuse inside a single Multi-Stream Action Transformer trained
  with flow matching. Memory, multimodal perception, and physical
  conditioning slot into the same backbone instead of bolt-on heads.
- **Perception and Memory via cognition tokens.** A small set of learnable cognition
  tokens are injected into historical observations and routed through a
  Qwen3-VL backbone, so the policy consumes a compact perceptual summary
  rather than raw VLM hidden states.
- **History awareness.** A Transformer memory module aggregates the
  per-timestep cognition tokens across history, giving the action model
  long-horizon context for tasks that require it.
- **Multimodal perception.** Visual, tactile, and torque inputs join the
  MSAT as a dedicated stream with 3-way joint attention; the model also
  predicts future physical signals via flow matching as an auxiliary
  objective.
- **Agile real-time control.** Asynchronous action execution and chunk-
  level inference optimization keep latency low enough for high-DoF
  control loops on real hardware.

---

## Performance

### Simulation Benchmarks

Success rates (%) of RLDX-1 fine-tuned on each benchmark's training set.

| Benchmark | Success Rate | Fine-tuned Checkpoint |
|-----------|--------------|------------------------|
| LIBERO (Avg) | 97.8 | [RLWRLD/RLDX-1-FT-LIBERO](https://huggingface.co/RLWRLD/RLDX-1-FT-LIBERO) |
| LIBERO-Plus | 87.6 | [RLWRLD/RLDX-1-FT-LIBERO](https://huggingface.co/RLWRLD/RLDX-1-FT-LIBERO) |
| SIMPLER Google-VM | 81.5 | [RLWRLD/RLDX-1-FT-SIMPLER-GOOGLE](https://huggingface.co/RLWRLD/RLDX-1-FT-SIMPLER-GOOGLE) |
| SIMPLER Google-VA | 77.4 | [RLWRLD/RLDX-1-FT-SIMPLER-GOOGLE](https://huggingface.co/RLWRLD/RLDX-1-FT-SIMPLER-GOOGLE) |
| SIMPLER WidowX | 71.9 | [RLWRLD/RLDX-1-FT-SIMPLER-WIDOWX](https://huggingface.co/RLWRLD/RLDX-1-FT-SIMPLER-WIDOWX) |
| RoboCasa Kitchen (24 tasks) | 70.6 | [RLWRLD/RLDX-1-FT-ROBOCASA](https://huggingface.co/RLWRLD/RLDX-1-FT-ROBOCASA) |
| GR-1 Tabletop | 58.7 | [RLWRLD/RLDX-1-FT-GR1](https://huggingface.co/RLWRLD/RLDX-1-FT-GR1) |
| RoboCasa365 (Avg) | 31.5 | [RLWRLD/RLDX-1-FT-RC365](https://huggingface.co/RLWRLD/RLDX-1-FT-RC365) |

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
| [`training.md`](docs/training.md) | `launch_train.py` CLI, finetune / pre-train / mid-train recipes, image-pipeline knobs |
| [`evaluation.md`](docs/evaluation.md) | RoboCasa / LIBERO / SIMPLER / GR-1 eval, server + rollout split, results aggregation |
| [`inference_server.md`](docs/inference_server.md) | `run_rldx_server.py` CLI, wire protocol, simulator + real-robot deployment |


---

## Checkpoints

| Checkpoint | Description | Params | HuggingFace |
|-----------|-------------|--------|-------------|
| `RLDX-1-PT` | Pre-trained (video) | 6.9B | [RLWRLD/RLDX-1-PT](https://huggingface.co/RLWRLD/RLDX-1-PT) |
| `RLDX-1-FT-ROBOCASA` | Finetuned on RoboCasa | 6.9B | [RLWRLD/RLDX-1-FT-ROBOCASA](https://huggingface.co/RLWRLD/RLDX-1-FT-ROBOCASA) |
| `RLDX-1-MT-DROID` | Mid-trained on DROID with all add-ons | 8.1B | [RLWRLD/RLDX-1-MT-DROID](https://huggingface.co/RLWRLD/RLDX-1-MT-DROID) |
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

Pass it via `--modality-config-path my_modality_config.py` during training.

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

### With All Add-ons (Memory + Motion + Physics)

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

---

## Reproducing Benchmark Results

Each benchmark ships with a self-contained README covering setup, the
HuggingFace checkpoint, and evaluation commands.

| Benchmark | Embodiment Tag | Guide |
|---|---|---|
| LIBERO | `GENERAL_EMBODIMENT` | [`run_scripts/eval/libero/README.md`](run_scripts/eval/libero/README.md) |
| LIBERO-Plus | `GENERAL_EMBODIMENT` | [`run_scripts/eval/libero_plus/README.md`](run_scripts/eval/libero_plus/README.md) |
| SimplerEnv (Google / WidowX) | `OXE_BRIDGE_ORIG` | [`run_scripts/eval/simpler/README.md`](run_scripts/eval/simpler/README.md) |
| GR-1 Tabletop | `GENERAL_EMBODIMENT` | [`run_scripts/eval/gr1_tabletop/README.md`](run_scripts/eval/gr1_tabletop/README.md) |
| RoboCasa Kitchen (24 tasks) | `GENERAL_EMBODIMENT` | [`run_scripts/eval/robocasa_kitchen/README.md`](run_scripts/eval/robocasa_kitchen/README.md) |
| RoboCasa365 | `GENERAL_EMBODIMENT` | [`run_scripts/eval/robocasa_365/README.md`](run_scripts/eval/robocasa_365/README.md) |

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
@article{rldx2025,
  title={RLDX-1: A Unified Vision-Language-Action Model with Modular Temporal Reasoning},
  author={TBD},
  journal={arXiv preprint arXiv:TBD},
  year={2026}
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
