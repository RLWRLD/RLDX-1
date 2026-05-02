# RoboCasa365

Large-scale household manipulation benchmark extending RoboCasa Kitchen
to a broader distribution of everyday tasks. Reported success rate
**31.5 %** (Avg over Atomic-Seen / Composite-Seen / Composite-Unseen).

| Field | Value |
|---|---|
| Embodiment tag | `GENERAL_EMBODIMENT` |
| HuggingFace checkpoint | unreleased — tracked via [#75](https://github.com/RLWRLD/RLDX/issues/75) |
| Simulator venv | `rldx/eval/sim/robocasa365/robocasa365_uv/.venv` |

## 1. Setup (one-time)

```bash
bash run_scripts/eval/robocasa_365/setup_robocasa365.sh
```

Creates the standalone simulator venv and downloads RoboCasa365 assets.

## 2. Finetune from RLDX-1-PT

```bash
DATA_DIR=/path/to/robocasa365/v1.0/pretrain \
bash run_scripts/train/benchmarks/finetune_rldx1_robocasa_365.sh
```

Override `BASE_MODEL_PATH=...` to start from a different base.

## 3. Run evaluation

```bash
bash run_scripts/eval/robocasa_365/eval_robocasa365.sh \
    --model-path <RC365 checkpoint path or HF repo> \
    --task-set target50 \
    --split target
```

Key flags:

| Flag | Default | Notes |
|---|---|---|
| `--model-path` | required | HF repo or local checkpoint dir |
| `--task-set` | `target50` | `atomic_seen` / `composite_seen` / `composite_unseen` / `target50` |
| `--split` | `target` | `pretrain` (training-distribution layouts) or `target` (held-out) |
| `--n-episodes` | 50 | Episodes per task |
| `--n-envs` | 5 | Parallel sim environments |
| `--n-action-steps` | 8 | Action chunk size |
| `--num-shards` / `--shard-index` | 1 / 0 | Optional sharding for parallel runs |

Outputs land in `output/robocasa365_eval/<exp_name>/`.

Per-task step horizons are defined in
[`task_sets.yaml`](task_sets.yaml). Sections:

- `atomic_seen` — atomic tasks included during pretraining
- `composite_seen` — composite tasks included during pretraining
- `composite_unseen` — composite tasks **not** in pretraining (OOD)
- `target50` — union of the three above (full benchmark)
