# Training

RLDX has a single unified launcher — `rldx/experiment/launch_train.py` —
that covers pre-training, mid-training, and finetuning. The difference
between the three is just which dataset-selection arguments and which
base checkpoint you pass.

This doc covers the CLI surface, the three common training modes, and
where to look for canonical example scripts.

## Entry point

```bash
uv run torchrun --nproc_per_node=<N_GPUS> --master_port=<PORT> \
    rldx/experiment/launch_train.py \
        [ CLI args here ]
```

`launch_train.py` parses a single `TrainConfig` dataclass
(`rldx/configs/train_config.py`) via
[`tyro`](https://github.com/brentyi/tyro) — every field on that
dataclass is exposed as a CLI flag with the same name, with underscores
turned into dashes (e.g. `global_batch_size` → `--global-batch-size`).

The launcher internally reuses `run()` from
`rldx/experiment/experiment.py`, which handles dataset construction,
model setup, DeepSpeed initialisation, and the HuggingFace `Trainer`
main loop.

## Three training modes

### (a) Finetune from a base checkpoint

The common case: take a pre-trained RLDX checkpoint and specialise it
on a single dataset.

```bash
uv run torchrun --nproc_per_node=8 rldx/experiment/launch_train.py \
    --base-model-path RLWRLD/RLDX-1-PT \
    --dataset-path /path/to/robocasa_dataset \
    --embodiment-tag GENERAL_EMBODIMENT \
    --modality-config-path rldx/configs/data/robocasa_config.py \
    --video-length 4 \
    --n-cog-tokens 64 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --global-batch-size 64 \
    --learning-rate 1e-4 \
    --max-steps 60000 \
    --save-steps 1000 \
    --save-total-limit 5 \
    --output-dir ./ckpt/rldx/finetuned/my_run \
    --experiment-name my_run \
    --use-wandb --wandb-project rldx-finetune
```

Key finetune flags:

| Flag | Purpose |
|---|---|
| `--base-model-path` | HF hub id or local path to the pre-trained checkpoint. Processor + config + weights are all loaded from it. |
| `--dataset-path` | LeRobot-format dataset directory (single-dataset finetune). |
| `--dataset-paths` + `--dataset-mix-ratios` | Multi-dataset finetune with per-dataset mix ratios. Mutually exclusive with `--dataset-path`. |
| `--embodiment-tag` | Which embodiment slot to train. `GENERAL_EMBODIMENT` routes through the category-specific MLP's "new" head. |
| `--modality-config-path` | Python file declaring the modality (video / state / action) config for this embodiment. See [`rldx/configs/data/`](../rldx/configs/data/) for the built-ins. |

Canonical script:
[`run_scripts/train/finetune_rldx_robocasa.sh`](../run_scripts/train/finetune_rldx_robocasa.sh).

### (b) Pretrain from scratch (or from Qwen3-VL base)

Pretraining consumes a **dataset mix** — a named collection of
LeRobot datasets declared in
[`rldx/configs/data/dataset_mix.py`](../rldx/configs/data/dataset_mix.py)
(`rldx_mix_v4`, `rldx_mix_v5`, etc.) — rather than a single directory.

```bash
uv run torchrun --nproc_per_node=8 rldx/experiment/launch_train.py \
    --pt-dataset-root /data/pre-train \
    --pt-dataset-mix rldx_mix_v4 \
    --video-length 4 \
    --n-cog-tokens 64 \
    --global-batch-size 1024 \
    --learning-rate 1e-4 \
    --max-steps 100000 \
    --save-steps 5000 \
    --output-dir ./ckpt/rldx/pre-trained/my_pretrain
```

Key pre-train flags:

| Flag | Purpose |
|---|---|
| `--pt-dataset-root` | Root directory containing every dataset referenced by the mix. |
| `--pt-dataset-mix` | Name of a mix registered in `dataset_mix.py`. |

`--pt-*` flags are mutually exclusive with `--dataset-path` /
`--dataset-paths` — pick one of the two modes.

Canonical scripts:
[`run_scripts/train/pretrain/pretrain_rldx1_mixv4_multinode.sh`](../run_scripts/train/pretrain/pretrain_rldx1_mixv4_multinode.sh)
and `pretrain_rldx1_oxe_multinode.sh` in the same directory.

### (c) Mid-train with optional add-ons (memory / motion / physics)

Mid-training resumes from a pre-trained checkpoint and additionally
turns on one or more add-on components for supervised continuation. All
three add-ons are additive flags on top of the base finetune / pre-train
command lines:

```bash
# Memory
--use-memory \
--memory-length 4 \
--memory-n-cog-tokens 16 \
--concat-memory \
--memory-dropout-prob 0.3

# Motion module
--use-motion \
--motion-insert-layer 9 \
--motion-injection-point vision_encoder \
--motion-pool-type avg

# Physics (tactile + torque)
--use-physics \
--physics-keys tactile torque \
--physics-dims 30 7 \
--physics-loss-weight 0.1
```

All three can be combined in a single run. Canonical scripts:
[`run_scripts/train/ablations/`](../run_scripts/train/ablations/).

## Image pipeline flags

See [`architecture.md`](architecture.md#stage-1--image-pipeline) for the
geometry. The three knobs are:

```bash
--image-max-area 65536       # area budget, default 256*256
--image-resize-m 32          # alignment multiple
--random-crop-fraction 0.9   # None (default) = no-op
--random-rotation-angle 5    # optional train-only rotate
--color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
```

`--random-crop-fraction=None` is the production default — the refactor
confirmed that the current datasets are 256×256 and the crop is
skipped. Set it only when you want extra augmentation.

## Training knobs cheat-sheet

| Flag | Default | Notes |
|---|---|---|
| `--num-gpus` | 1 | Also needs matching `torchrun --nproc_per_node`. |
| `--global-batch-size` | 64 | Divided by `num_gpus * gradient_accumulation_steps`. |
| `--gradient-accumulation-steps` | 1 | |
| `--learning-rate` | 1e-4 | Warmup via `--warmup-ratio` (default 0.05). |
| `--lr-scheduler-type` | cosine | |
| `--weight-decay` | 1e-5 | |
| `--max-steps` | 10000 | Total training steps. |
| `--save-steps` | 1000 | Checkpoint cadence. |
| `--save-total-limit` | 5 | Rolling window of saved checkpoints. |
| `--dataloader-num-workers` | 2 | Workers per rank. |
| `--use-wandb` + `--wandb-project` | off | `WANDB_PROJECT` env var is also honored. |
| `--experiment-name` | "debug" | Run name on wandb and for the output directory. |

## Tuning which parameters are trained

Defaults reflect the refactor decision: backbone frozen below, action
head fully tuned.

| Flag | Default | What it controls |
|---|---|---|
| `--tune-llm` | False | Train the entire Qwen3-VL LM backbone. |
| `--tune-visual` | False | Train the vision tower. |
| `--tune-top-llm-layers` | 4 | When `--tune-llm` is off, train just the top N LM layers. |
| `--tune-projector` | True | Multi-modal projector. |
| `--tune-diffusion-model` | True | MSAT action model. |
| `--freeze-cog-tokens` | False | Freeze the learnable cognition-token embeddings. |
| `--state-dropout-prob` | 0.0 | Stochastic state dropout for regularisation. |
| `--general-embodiment-train-ratio` | 0 | Mix-in ratio of general-embodiment (cross-embodiment) samples when finetuning on a single embodiment. 0 disables the mix. |

## Checkpoint format

After every `save_steps`, the `CheckpointFormatCallback` writes the
trainer output plus:

```
{output_dir}/checkpoint-{step}/
├── config.json                   # model_type="RLDX", see rldx/configs/model/rldx.py
├── model-00001-of-00003.safetensors
├── ...
├── experiment_cfg/               # conf.yaml, final_model_config.json, dataset_statistics.json
├── processor/                    # <-- subdir layout required for inference
│   ├── processor_config.json
│   ├── embodiment_id.json
│   └── statistics.json
├── wandb_config.json
└── train.log
```

Consumers load checkpoints via `AutoProcessor.from_pretrained(ckpt/processor)`
(note the subdir). `RLDXPolicy` handles this automatically.

## Troubleshooting

### "RuntimeError: CUDA out of memory"
- Raise `--gradient-accumulation-steps` to keep global batch constant.
- Switch to ZeRO-3: the stage is read from `TrainingConfig.deepspeed_stage` (nested; not a top-level CLI flag). Flip it to `3` in your launcher or the defaults in `rldx/configs/training/training_config.py`, which dispatches to `rldx/configs/deepspeed/zero3_config.json`.
- Drop `--tune-top-llm-layers` to 0 to freeze more of the backbone.

### "Loaded memory_length=1 from checkpoint" when you expected more
The base checkpoint was trained without memory; `--use-memory` alone
is not enough, you also need `--memory-length`, `--memory-n-cog-tokens`,
and usually `--concat-memory`. Double-check that the base model actually
supports the memory path (all current RLDX pre-training runs do).

### "processor_config.json not found" at checkpoint load
The base checkpoint was uploaded with the legacy flat layout. Either
move its processor files into a `processor/` subdir and re-upload, or
apply `scripts/patch_checkpoint.py` to fix up an existing directory.
All new checkpoints saved by the `CheckpointFormatCallback` already use
the subdir layout.

## Where to next

- [`installation.md`](installation.md) — environment setup
- [`evaluation.md`](evaluation.md) — run the trained checkpoint on benchmarks
- [`inference_server.md`](inference_server.md) — serve checkpoints over ZeroMQ
- [`architecture.md`](architecture.md) — what the config flags are actually wiring up
