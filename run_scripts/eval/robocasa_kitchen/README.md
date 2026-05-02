# RoboCasa Kitchen (24 tasks)

Single-arm kitchen manipulation on a mobile Panda manipulator
(`PandaOmron`). 24 tasks: pick-and-place, doors, drawers, appliances.
Two fixed external cameras (left/right) + one wrist camera, 256×256.

| Field | Value |
|---|---|
| Embodiment tag | `GENERAL_EMBODIMENT` |
| HuggingFace checkpoint | [`RLWRLD/RLDX-1-FT-ROBOCASA`](https://huggingface.co/RLWRLD/RLDX-1-FT-ROBOCASA) |
| Reported success rate | 70.6 % |
| Simulator venv | `rldx/eval/sim/robocasa/robocasa_uv/.venv` |

## 1. Setup (one-time)

```bash
bash run_scripts/eval/robocasa_kitchen/setup_robocasa.sh
```

Builds the simulator venv and applies patches under
`run_scripts/eval/robocasa_kitchen/patches/` (notably
`seed_clamp_64bit.patch` to fix `gymnasium 0.29` 64-bit seed forwarding).

## 2. Finetune from RLDX-1-PT

300-demo (release) recipe — matches the technical-report number:

```bash
DATA_DIR=/path/to/robocasa_mg_gr00t_300 \
bash run_scripts/train/benchmarks/finetune_rldx1_robocasa_kitchen300_lr4_bs1024.sh
```

A 1000-demo variant is also provided
(`finetune_rldx1_robocasa_kitchen1000_lr4_bs1024.sh`).

## 3. Run evaluation

```bash
bash run_scripts/eval/robocasa_kitchen/eval_robocasa.sh \
    RLWRLD/RLDX-1-FT-ROBOCASA
```

Parallel 4-GPU runner that shards the 24 tasks across local GPUs and runs
50 episodes per task. Outputs land in
`output_final/robocasa/<ckpt>/<task>/`.
