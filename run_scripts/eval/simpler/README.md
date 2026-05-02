# SimplerEnv (Google Robot + WidowX)

Single-arm gripper benchmarks built on SimplerEnv. Two robot families
share the same simulator and venv but use different fine-tuned heads.

| Variant | Robot | Tasks | Embodiment tag | HuggingFace checkpoint |
|---|---|---|---|---|
| **SIMPLER Google** | Google robot (Fractal data) | 4 (Move Near, Pick Coke Can, Open/Close Drawer) | `OXE_BRIDGE_ORIG` | [`RLWRLD/RLDX-1-FT-SIMPLER-GOOGLE`](https://huggingface.co/RLWRLD/RLDX-1-FT-SIMPLER-GOOGLE) |
| **SIMPLER WidowX** | WidowX (Bridge data) | 4 (Spoon-on-Towel, Carrot-on-Plate, Stack Cube, Eggplant-in-Basket) | `OXE_BRIDGE_ORIG` | [`RLWRLD/RLDX-1-FT-SIMPLER-WIDOWX`](https://huggingface.co/RLWRLD/RLDX-1-FT-SIMPLER-WIDOWX) |

Reported success rates: Google-VM **81.5 %**, Google-VA **77.4 %**, WidowX **71.9 %**.

Simulator venv: `rldx/eval/sim/SimplerEnv/simpler_uv/.venv`.

## 1. Setup (one-time)

```bash
bash run_scripts/eval/simpler/setup_simpler.sh
```

Builds the shared SimplerEnv venv and clones `ManiSkill2_real2sim`.

## 2. Finetune from RLDX-1-PT

```bash
# Google (Fractal)
DATA_DIR=/path/to/fractal20220817 \
bash run_scripts/train/benchmarks/finetune_simpler_google_bs1024_sd05.sh

# WidowX (Bridge)
DATA_DIR=/path/to/bridge_orig \
bash run_scripts/train/benchmarks/finetune_simpler_widowx_bs1024.sh
```

Both default to `RLWRLD/RLDX-1-PT` as the base; override with
`BASE_MODEL_PATH=...`.

## 3. Run evaluation

```bash
# Google (Visual Matching)
bash run_scripts/eval/simpler/eval_simpler.sh google_vm RLWRLD/RLDX-1-FT-SIMPLER-GOOGLE

# Google (Variant Aggregation — 4 tasks × 7 variants = 28 environments)
bash run_scripts/eval/simpler/eval_simpler.sh google_va RLWRLD/RLDX-1-FT-SIMPLER-GOOGLE

# WidowX (Bridge)
bash run_scripts/eval/simpler/eval_simpler.sh widowx    RLWRLD/RLDX-1-FT-SIMPLER-WIDOWX
```

Outputs land in `output_final/simpler_<variant>/<ckpt>/<task>/`. The
`MODEL_PATH` argument is optional and defaults to the matching released
checkpoint per variant.
