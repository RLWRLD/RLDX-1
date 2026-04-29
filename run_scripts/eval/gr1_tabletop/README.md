# GR-1 Tabletop

GR-1 humanoid tabletop manipulation (24 tasks: 18 object rearrangement +
6 articulated object). Ego-centric camera, 256×256 resolution.

| Field | Value |
|---|---|
| Embodiment tag | `GR1` |
| HuggingFace checkpoint | [`RLWRLD/RLDX-1-FT-GR1`](https://huggingface.co/RLWRLD/RLDX-1-FT-GR1) |
| Reported success rate | 58.7 % (24-task average) |
| Simulator venv | `rldx/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv` |

## 1. Setup (one-time)

```bash
bash run_scripts/eval/gr1_tabletop/setup_gr1.sh
```

Initializes the simulator submodule and applies patches under
`run_scripts/eval/gr1_tabletop/patches/` (e.g. seed clamping for the
gymnasium 0.29 vector-env layer).

## 2. Finetune from RLDX-1-PT

```bash
DATA_ROOT=/path/to/GR00T-X-Embodiment-Sim \
bash run_scripts/train/benchmarks/finetune_gr1_tabletop_bs1024_sd05.sh
```

Defaults to `RLWRLD/RLDX-1-PT` as the base; override with
`BASE_MODEL_PATH=...`.

## 3. Run evaluation

```bash
bash run_scripts/eval/gr1_tabletop/eval_gr1.sh RLWRLD/RLDX-1-FT-GR1
```

Loops over the 24 GR-1 tabletop tasks sequentially against a single model
server. Outputs land in `output_final/gr1_tabletop/<ckpt>/<task>/`.
