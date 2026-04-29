# Evaluation

This doc covers how to run a trained RLDX checkpoint on each of the
simulator benchmarks the repo ships with, plus the general
server/client architecture those evals use.

## Architecture (why there are two processes)

RLDX evaluation splits cleanly into two processes:

```
  ┌─────────────────────────┐       ZeroMQ         ┌───────────────────────┐
  │  run_rldx_server.py     │ ───────────────────▶ │  simulator rollout    │
  │  (training env .venv)   │ ◀─────────────────── │  (per-sim eval venv)  │
  │  loads RLDX, holds GPU  │   observations /     │  mujoco/robosuite/... │
  │  answers policy queries │   actions             │  no RLDX imports      │
  └─────────────────────────┘                      └───────────────────────┘
```

The **model server** lives in the main training venv (`.venv/`), loads
the RLDX checkpoint onto the GPU, and listens on a TCP port via ZeroMQ.
The **rollout process** lives in a **simulator-specific uv venv** (for
example `rldx/eval/sim/robocasa/robocasa_uv/.venv`), steps the
environment, serialises observations to the server, and applies the
returned action chunks. This split exists because the mujoco/robosuite
stacks conflict with the `torch==2.7 + flash-attn` pins required by
training, and keeping them separated means neither env has to compromise.

Every benchmark below follows the same pattern: (1) launch the model
server, (2) launch the rollout worker against that server, (3) the
rollout writes videos + a CSV summary under `output_final/{benchmark}/`.

## Shared flags

### Model server — `rldx/eval/run_rldx_server.py`

| Flag | Default | Notes |
|---|---|---|
| `--model-path` | *(required)* | HF hub id or local checkpoint path. The processor is loaded from `{model_path}/processor` automatically. |
| `--embodiment-tag` | `GENERAL_EMBODIMENT` | Which category-specific encoder/decoder head to use. See `rldx/data/embodiment_tags.py` for the full list. |
| `--use-sim-policy-wrapper` | False | Wraps `RLDXPolicy` in `RLDXSimPolicyWrapper`, which reshapes observations and action chunks to the layout simulator eval expects. **Always on for sim eval.** |
| `--host` / `--port` | `127.0.0.1 / 5555` | Where the server listens. `5555` is `DEFAULT_MODEL_SERVER_PORT` in `run_rldx_server.py`; the examples below use an arbitrary port like `20100` when running multiple servers on one host. |
| `--deactivate-memory` | False | Loads a memory-trained checkpoint as non-memory model for the "ablation without memory" study. |
| `--sample-timestep-from-beta-dist` | False | Use a Beta-distributed denoising timestep instead of uniform. |
| `--denoising-timesteps` | None | Override the default `num_inference_timesteps = 4`. |

### Rollout runner — `rldx/eval/rollout_policy.py`

| Flag | Default | Notes |
|---|---|---|
| `--env-name` | *(required)* | Gym env id. Format depends on the benchmark (see below). |
| `--n-episodes` | 50 | Total episodes to run. Split across `n_envs`. |
| `--n-envs` | 8 | Vectorised sim envs. >1 requires robosuite to support async; set to 1 if unsure. |
| `--n-action-steps` | 8 | Execution horizon per action chunk (must be ≤ `action_horizon`). |
| `--max-episode-steps` | 504 | Max steps per episode before truncation. |
| `--policy-client-host/port` | *(required)* | Must match the server. |
| `--video-dir` | *(required)* | Output directory for `.mp4` files and `simulation_results.csv`. Resume logic auto-skips already-recorded episodes in the same dir. |

## RoboCasa kitchen (24 tasks)

Canonical SLURM script:
[`run_scripts/eval/robocasa_kitchen/eval_robocasa.sh`](../run_scripts/eval/robocasa_kitchen/eval_robocasa.sh)

Canonical local parallel launcher (4 GPUs, no SLURM):
[`run_scripts/eval/robocasa_kitchen/eval_robocasa.sh`](../run_scripts/eval/robocasa_kitchen/eval_robocasa.sh)

Single-task example:

```bash
# Terminal 1 — model server
uv run python rldx/eval/run_rldx_server.py \
    --model-path RLWRLD/RLDX-1-FT-ROBOCASA \
    --embodiment-tag GENERAL_EMBODIMENT \
    --use-sim-policy-wrapper \
    --host 127.0.0.1 --port 20100

# Terminal 2 — rollout in the robocasa venv
rldx/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
    rldx/eval/rollout_policy.py \
        --n-episodes 50 \
        --policy-client-host 127.0.0.1 --policy-client-port 20100 \
        --max-episode-steps 720 \
        --env-name "robocasa_panda_omron/TurnSinkSpout_PandaOmron_Env" \
        --n-action-steps 16 \
        --n-envs 1 \
        --video-dir output_final/robocasa/my_ckpt/TurnSinkSpout
```

The 24 kitchen tasks are:

```
TurnSinkSpout       TurnOnStove        TurnOnSinkFaucet    TurnOnMicrowave
TurnOffStove        TurnOffSinkFaucet  TurnOffMicrowave
PnPStoveToCounter   PnPSinkToCounter   PnPMicrowaveToCounter
PnPCounterToStove   PnPCounterToSink   PnPCounterToMicrowave
PnPCounterToCab     PnPCabToCounter
OpenSingleDoor      OpenDrawer         OpenDoubleDoor
CoffeeSetupMug      CoffeeServeMug     CoffeePressButton
CloseSingleDoor     CloseDrawer        CloseDoubleDoor
```

Each one is a separate gym env id of the form
`robocasa_panda_omron/{task}_PandaOmron_Env`. The 4-GPU launcher runs 6
tasks per GPU sequentially and aggregates results under
`output_final/robocasa/{CKPT_NAME}/{TASK_NAME}/`.

## RoboCasa 365 (larger task set)

[`run_scripts/eval/robocasa_365/`](../run_scripts/eval/robocasa_365/)
bundles a SLURM submission flow:

```bash
sbatch run_scripts/eval/robocasa_365/submit_eval_rldx_bsz128_rc365.slurm.sh
```

Config defaults live in
`submit_eval_rldx_bsz128_rc365.config.json`.
`task_sets.yaml` lists the task splits. After every job finishes, run
`upload_merged_eval_to_wandb.py` to push the aggregated metrics to
Weights & Biases.

## LIBERO (4 suites × 10 tasks)

Canonical script:
[`run_scripts/eval/libero/eval_rldx_libero.sh`](../run_scripts/eval/libero/eval_rldx_libero.sh)

Setup the sim env once:

```bash
bash rldx/eval/sim/LIBERO/setup_libero.sh
```

LIBERO env ids look like `libero_{suite}_{task_idx}`, for example
`libero_spatial_0`. The script array-indexes across all 40 tasks (10
per suite × 4 suites). Same model-server + rollout split as RoboCasa,
just with the LIBERO venv at
`rldx/eval/sim/LIBERO/libero_uv/.venv/bin/python`.

## CALVIN long-horizon

Canonical script:
[`run_scripts/eval/calvin/eval_calvin_1d4_iter.sh`](../run_scripts/eval/calvin/eval_calvin_1d4_iter.sh)

CALVIN evaluates 1000 task chains (5 sub-tasks each). Entry point is
`rldx/eval/sim/CALVIN/eval_taskwise.py` — this script does **not** use
`rollout_policy.py`; instead it runs its own per-task loop because
CALVIN's sub-task chaining is different from the vec-env rollout
pattern.

## GR1 tabletop (humanoid)

[`run_scripts/eval/gr1_tabletop/`](../run_scripts/eval/gr1_tabletop/):

```bash
# Setup (one-time)
bash run_scripts/eval/gr1_tabletop/setup_gr1.sh

# Single task
bash run_scripts/eval/gr1_tabletop/eval_gr1.sh my_checkpoint_path

# Multi-task sweep
bash run_scripts/eval/gr1_tabletop/eval_gr1_multi.sh my_checkpoint_path
```

GR1 uses robocasa-gr1-tabletop-tasks with the Unitree G1 embodiment.
Pass `--embodiment-tag UNITREE_G1` to the server.

## Collecting results

Every rollout writes per-episode `.mp4` files with either `-success` or
`-failure` in the filename, plus a `simulation_results.csv` summary.
Quick success-rate tally:

```bash
cd output_final/robocasa/{ckpt}/{task}
S=$(ls *-success.mp4 2>/dev/null | wc -l)
F=$(ls *-failure.mp4 2>/dev/null | wc -l)
echo "$S / $((S+F)) ($((100*S/(S+F)))%)"
```

For multi-task aggregation:

```bash
for d in output_final/robocasa/{ckpt}/*/; do
    task=$(basename "$d")
    [ "$task" = "_launcher_logs" ] && continue
    s=$(ls "$d"*-success.mp4 2>/dev/null | wc -l)
    f=$(ls "$d"*-failure.mp4 2>/dev/null | wc -l)
    total=$((s+f))
    [ $total -gt 0 ] && printf "%-30s %2d/%2d (%3d%%)\n" "$task" "$s" "$total" "$((100*s/total))"
done | sort
```

## Troubleshooting

### Server loads but rollout gets `Connection refused`

The server takes 30-60 s to finish loading the backbone. The rollout
launcher included in
[`eval_robocasa_4gpu_local.sh`](../run_scripts/eval/robocasa_kitchen/eval_robocasa.sh)
polls `ss -lnt` until the port is ready; if you are writing your own
launcher, add a similar 60-120 s wait before the first rollout.

### Rollouts crash with `ValueError: Unrecognized processing class`

The checkpoint processor files are at the repo root instead of the
`processor/` subdir. Either re-save the checkpoint so the layout matches
the new format, or run `scripts/patch_checkpoint.py` on the local copy.

### `HF_HOME=/nonexistent` on a cluster

Some clusters intentionally point `HF_HOME` at `/dev/null` equivalents.
The server process will fail to cache the backbone download. Set
`HF_HOME` to a writable scratch path before launching, or pre-download
the backbone once and pass a local path via `--model-path`.

### Resumption — "I already ran half of the episodes"

`rollout_policy.py` scans the video directory on startup and counts
existing `episode_*.mp4` files per env index, then skips forward to the
next unrecorded episode. Just rerun the same command with the same
`--video-dir`.

## Where to next

- [`inference_server.md`](inference_server.md) — running the same
  `run_rldx_server.py` for real-robot deployment
- [`training.md`](training.md) — produce the checkpoint you are
  evaluating
- [`architecture.md`](architecture.md) — what the model does under the
  hood
