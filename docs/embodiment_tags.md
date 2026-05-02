# Embodiment Tags

Every dataset and every inference call goes through an `EmbodimentTag`.
The tag selects a slot in the
[category-specific MLP encoder/decoder](architecture.md#stage-3--state--action-stream)
— a per-embodiment pair of small MLPs that maps the robot's raw state
and action vectors into the model's shared latent space.

The naming convention and the list itself originate from
[NVIDIA Isaac GR00T v1.7](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.7-release);
RLDX preserves and extends it. See
[`rldx/data/embodiment_tags.py`](../rldx/data/embodiment_tags.py)
for the canonical enum.

## Picking a tag for a new robot

| Your case | Tag to use |
|---|---|
| Custom robot, finetuning on top of `RLDX-1-PT` | `GENERAL_EMBODIMENT` |
| Re-running an OXE benchmark with the original dataset key naming | The matching `OXE_*` tag (e.g. `OXE_FRACTAL`, `OXE_DROID`) |
| Reproducing one of the listed simulation benchmarks | Tag listed in [Reproducing Benchmark Results](../README.md#reproducing-benchmark-results) |
| Migrating a checkpoint that used the old `NEW_EMBODIMENT` slot | `GENERAL_EMBODIMENT` (post-PR #10 rename, kept as a legacy alias) |

`GENERAL_EMBODIMENT` is always the safe default. It is the slot that
`RLDX-1-PT` reserves for downstream finetuning, so any custom robot's
state/action MLP heads land there.

## Tag categories

| Group | Examples | When it applies |
|---|---|---|
| `GENERAL_EMBODIMENT` | — | Single slot for downstream / new robots |
| `OXE_*` | `OXE_FRACTAL`, `OXE_BRIDGE_ORIG`, `OXE_DROID`, ... | Open-X Embodiment datasets used in pre-training |
| Sim / benchmark | `LIBERO_PANDA`, `ROBOCASA_PANDA_OMRON`, `OXE_GOOGLE`, `OXE_WIDOWX`, `GR1` | Specific simulation benchmark embodiments |
| Humanoid / dexterous | `AGIBOT_DEXHAND`, `HUMANOID_EVERYDAY_G1`, `UNITREE_G1`, `BEHAVIOR_R1_PRO`, `NEURAL_GR1` | Pre-training datasets with their own joint conventions |

The full enum and the dataset-key string each tag maps to are in
[`rldx/data/embodiment_tags.py`](../rldx/data/embodiment_tags.py).

## How the tag is consumed

At training time the tag determines:

1. Which row of the embodiment-conditioned MLP gets gradient updates
   (each tag has its own input/output projection pair).
2. Which `MODALITY_CONFIGS` entry is loaded (in
   [`rldx/configs/data/embodiment_configs.py`](../rldx/configs/data/embodiment_configs.py))
   — this dictates the modality keys the data loader and processor use.
3. Which `embodiment_id` integer the model sees as a conditioning
   token.

At inference time the loaded checkpoint already has trained weights
for the tag you trained with. The server resolves the tag's
`modality_keys` to slice incoming observations and to build the
action chunk it returns.

## Common pitfall: tag mismatch

A wrong `--embodiment-tag` rarely surfaces as a shape error. Instead
the lookup `state_action_processor.modality_configs[tag]["action"]`
silently returns the wrong embodiment's modality keys, and the
incoming observation dict is sliced under the wrong joint convention.
Symptoms include:

- `KeyError` on a modality key that the model expects but the dataset
  does not provide (or vice versa).
- The server appears to run, but the action chunk is garbage / nonsense
  joint values.

If the loaded checkpoint was finetuned on `GENERAL_EMBODIMENT`,
you must pass `--embodiment-tag GENERAL_EMBODIMENT` at inference time.
Production checkpoint → tag mappings live in the per-checkpoint memory
notes inside this repo's `.claude/memory/`.

## Where to next

- [`training.md`](training.md) — register a custom modality config
  alongside a tag
- [`inference_server.md`](inference_server.md) — `--embodiment-tag`
  on the server CLI
- [GR00T v1.7 README](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.7-release)
  — upstream concept and the original tag introduction
