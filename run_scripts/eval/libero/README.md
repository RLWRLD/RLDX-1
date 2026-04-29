# LIBERO

Single-arm tabletop manipulation on a Franka Panda robot with a parallel
gripper (40 tasks across Spatial / Object / Goal / Long suites).

| Field | Value |
|---|---|
| Embodiment tag | `GENERAL_EMBODIMENT` |
| HuggingFace checkpoint | [`RLWRLD/RLDX-1-FT-LIBERO`](https://huggingface.co/RLWRLD/RLDX-1-FT-LIBERO) |
| Reported success rate | 97.4 % (LIBERO Avg) / 84.3 % (LIBERO-Plus) |
| Simulator venv | `rldx/eval/sim/LIBERO/libero_uv/.venv` |

## 1. Setup (one-time)

```bash
bash run_scripts/eval/libero/setup_libero.sh
```

Builds an isolated `uv` venv for LIBERO and downloads task assets.

## 2. Finetune from RLDX-1-PT

```bash
DATA_DIR=/path/to/libero_delta \
bash run_scripts/train/benchmarks/finetune_libero_bs256.sh
```

Defaults to `RLWRLD/RLDX-1-PT` as the base; override with
`BASE_MODEL_PATH=...`.

## 3. Run evaluation

```bash
bash run_scripts/eval/libero/eval_libero.sh \
    libero_release \
    RLWRLD/RLDX-1-FT-LIBERO
```

Arguments: `<run_label>` (output dir name), `<MODEL_PATH>` (HF repo or
local checkpoint). Outputs land in
`output_final/libero/<run_label>/<suite>/<task>/`.
