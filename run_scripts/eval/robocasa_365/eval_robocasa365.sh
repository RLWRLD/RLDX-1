#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_scripts/eval/robocasa_365/eval_robocasa365.sh \
#     --model-path RLWRLD/RLDX-1-FT-RC365 \
#     --task-set target50 \
#     --split target

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PY365_DEFAULT="$PROJECT_REPO/rldx/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python"

MODEL_PATH=""
TASK_SET="target50"
SPLIT="target"
N_EPISODES=50
N_ENVS=5
N_ACTION_STEPS=8
MAX_EPISODE_STEPS=720
EMBODIMENT_TAG="GENERAL_EMBODIMENT"
SERVER_HOST="127.0.0.1"
SERVER_BIND_HOST="127.0.0.1"
SERVER_PORT=5555
SERVER_DEVICE="cuda"
PY365="${PY365:-$PY365_DEFAULT}"
OUTPUT_ROOT="$PROJECT_REPO/output/robocasa365_eval"
SERVER_WARMUP_SEC=20
TASK_YAML="$PROJECT_REPO/run_scripts/eval/robocasa_365/task_sets.yaml"
NUM_SHARDS=1
SHARD_INDEX=0
USER_SET_NUM_SHARDS=0
USER_SET_SHARD_INDEX=0
WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-robocasa365-eval}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_RESUME="${WANDB_RESUME:-allow}"
WANDB_LOG_VIDEOS="${WANDB_LOG_VIDEOS:-0}"
WANDB_MAX_VIDEOS_PER_TASK="${WANDB_MAX_VIDEOS_PER_TASK:-1}"
WANDB_MAX_VIDEO_MB="${WANDB_MAX_VIDEO_MB:-50}"
USE_TASK_HORIZON="${USE_TASK_HORIZON:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --task-set) TASK_SET="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --n-episodes) N_EPISODES="$2"; shift 2 ;;
    --n-envs) N_ENVS="$2"; shift 2 ;;
    --n-action-steps) N_ACTION_STEPS="$2"; shift 2 ;;
    --max-episode-steps) MAX_EPISODE_STEPS="$2"; shift 2 ;;
    --embodiment-tag) EMBODIMENT_TAG="$2"; shift 2 ;;
    --server-host) SERVER_HOST="$2"; shift 2 ;;
    --server-bind-host) SERVER_BIND_HOST="$2"; shift 2 ;;
    --server-port) SERVER_PORT="$2"; shift 2 ;;
    --server-device) SERVER_DEVICE="$2"; shift 2 ;;
    --python) PY365="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --server-warmup-sec) SERVER_WARMUP_SEC="$2"; shift 2 ;;
    --task-yaml) TASK_YAML="$2"; shift 2 ;;
    --num-shards) NUM_SHARDS="$2"; USER_SET_NUM_SHARDS=1; shift 2 ;;
    --shard-index) SHARD_INDEX="$2"; USER_SET_SHARD_INDEX=1; shift 2 ;;
    *)
      echo "[x] Unknown argument: $1"
      exit 1
      ;;
  esac
done

bool_to_int() {
  case "${1,,}" in
    1|true|yes|y|on) echo 1 ;;
    *) echo 0 ;;
  esac
}

WANDB_ENABLED_INT="$(bool_to_int "$WANDB_ENABLED")"
WANDB_LOG_VIDEOS_INT="$(bool_to_int "$WANDB_LOG_VIDEOS")"
USE_TASK_HORIZON_INT="$(bool_to_int "$USE_TASK_HORIZON")"

wandb_log_metrics_json() {
  local metrics_json="$1"
  local step="$2"
  if (( WANDB_ENABLED_INT == 0 )); then
    return 0
  fi

  WANDB_METRICS_JSON="$metrics_json" \
  WANDB_STEP="$step" \
  WANDB_PROJECT="$WANDB_PROJECT" \
  WANDB_ENTITY="$WANDB_ENTITY" \
  WANDB_RUN_ID="$WANDB_RUN_ID" \
  WANDB_RUN_NAME="$WANDB_RUN_NAME" \
  WANDB_RESUME="$WANDB_RESUME" \
  MODEL_PATH="$MODEL_PATH" \
  SPLIT="$SPLIT" \
  TASK_SET="$TASK_SET" \
  SHARD_INDEX="$SHARD_INDEX" \
  NUM_SHARDS="$NUM_SHARDS" \
  N_EPISODES="$N_EPISODES" \
  N_ENVS="$N_ENVS" \
  uv run --with wandb python - <<'PY' || true
import json
import os
import wandb

project = os.environ.get("WANDB_PROJECT", "robocasa365-eval")
entity = os.environ.get("WANDB_ENTITY") or None
run_id = os.environ.get("WANDB_RUN_ID") or None
run_name = os.environ.get("WANDB_RUN_NAME") or None
resume = os.environ.get("WANDB_RESUME", "allow")
step = int(os.environ.get("WANDB_STEP", "-1"))
metrics = json.loads(os.environ["WANDB_METRICS_JSON"])

run = wandb.init(
    project=project,
    entity=entity,
    id=run_id,
    name=run_name,
    resume=resume,
    reinit=True,
    tags=[
        "robocasa365",
        "eval",
        os.environ.get("SPLIT", "unknown"),
        os.environ.get("TASK_SET", "unknown"),
    ],
    config={
        "model_path": os.environ.get("MODEL_PATH"),
        "split": os.environ.get("SPLIT"),
        "task_set": os.environ.get("TASK_SET"),
        "num_shards": int(os.environ.get("NUM_SHARDS", "1")),
        "n_episodes": int(os.environ.get("N_EPISODES", "0")),
        "n_envs": int(os.environ.get("N_ENVS", "0")),
    },
)
if step >= 0:
    wandb.log(metrics, step=step)
else:
    wandb.log(metrics)
run.finish()
PY
}

wandb_log_artifact() {
  local artifact_name="$1"
  local artifact_type="$2"
  shift 2
  if (( WANDB_ENABLED_INT == 0 )); then
    return 0
  fi
  if [[ "$#" -eq 0 ]]; then
    return 0
  fi

  WANDB_PROJECT="$WANDB_PROJECT" \
  WANDB_ENTITY="$WANDB_ENTITY" \
  WANDB_RUN_ID="$WANDB_RUN_ID" \
  WANDB_RUN_NAME="$WANDB_RUN_NAME" \
  WANDB_RESUME="$WANDB_RESUME" \
  uv run --with wandb python - "$artifact_name" "$artifact_type" "$@" <<'PY' || true
import os
import sys
import wandb

project = os.environ.get("WANDB_PROJECT", "robocasa365-eval")
entity = os.environ.get("WANDB_ENTITY") or None
run_id = os.environ.get("WANDB_RUN_ID") or None
run_name = os.environ.get("WANDB_RUN_NAME") or None
resume = os.environ.get("WANDB_RESUME", "allow")

artifact_name = sys.argv[1]
artifact_type = sys.argv[2]
paths = sys.argv[3:]

run = wandb.init(
    project=project,
    entity=entity,
    id=run_id,
    name=run_name,
    resume=resume,
    reinit=True,
)
artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
added = 0
for p in paths:
    if os.path.isfile(p):
        artifact.add_file(p, name=os.path.basename(p))
        added += 1
if added > 0:
    run.log_artifact(artifact)
run.finish()
PY
}

if [[ -z "$MODEL_PATH" ]]; then
  echo "[x] --model-path is required"
  exit 1
fi

if [[ "$SPLIT" != "pretrain" && "$SPLIT" != "target" ]]; then
  echo "[x] --split must be one of: pretrain, target"
  exit 1
fi

if [[ ! -x "$PY365" ]]; then
  echo "[x] Python not found or not executable: $PY365"
  echo "[x] Run: bash run_scripts/eval/robocasa_365/setup_robocasa365.sh"
  exit 1
fi

cd "$PROJECT_REPO"
export NO_ALBUMENTATIONS_UPDATE=1

echo "[i] Project repo: $PROJECT_REPO"
echo "[i] Model path: $MODEL_PATH"
echo "[i] Task set: $TASK_SET"
echo "[i] Split: $SPLIT"
echo "[i] Episodes/task: $N_EPISODES"
echo "[i] Python: $PY365"
echo "[i] Task yaml: $TASK_YAML"

load_tasks_from_yaml_section() {
  local yaml_file="$1"
  local section="$2"
  awk -v section="$section" '
    /^[A-Za-z0-9_]+:[[:space:]]*$/ {
      key=$1
      sub(":", "", key)
      in_section = (key == section)
      next
    }
    in_section && /^[[:space:]]*-[[:space:]]*/ {
      line=$0
      sub(/^[[:space:]]*-[[:space:]]*/, "", line)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", line)
      if (length(line) > 0) print line
    }
  ' "$yaml_file"
}

load_task_horizons_from_yaml() {
  local yaml_file="$1"
  awk '
    /^task_horizons:[[:space:]]*$/ {
      in_section = 1
      next
    }
    in_section && /^[A-Za-z0-9_]+:[[:space:]]*$/ {
      # Reached another top-level section.
      exit
    }
    in_section && /^[[:space:]]+[A-Za-z0-9_]+:[[:space:]]*[0-9]+[[:space:]]*$/ {
      line = $0
      sub(/^[[:space:]]+/, "", line)
      split(line, kv, ":")
      key = kv[1]
      val = kv[2]
      gsub(/[[:space:]]/, "", val)
      print key "," val
    }
  ' "$yaml_file"
}

TASKS=()
if [[ -f "$TASK_YAML" && ( "$TASK_SET" == "atomic_seen" || "$TASK_SET" == "composite_seen" || "$TASK_SET" == "composite_unseen" || "$TASK_SET" == "target50" ) ]]; then
  if [[ "$TASK_SET" == "target50" ]]; then
    mapfile -t TASKS < <(
      {
        load_tasks_from_yaml_section "$TASK_YAML" "atomic_seen"
        load_tasks_from_yaml_section "$TASK_YAML" "composite_seen"
        load_tasks_from_yaml_section "$TASK_YAML" "composite_unseen"
      } | awk 'NF' | awk '!seen[$0]++'
    )
  else
    mapfile -t TASKS < <(load_tasks_from_yaml_section "$TASK_YAML" "$TASK_SET")
  fi
else
  echo "[x] Unknown task set: $TASK_SET"
  exit 1
fi

# Auto-wire sharding from Slurm array if shard args are omitted.
if [[ "$USER_SET_SHARD_INDEX" -eq 0 && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  SHARD_INDEX="$SLURM_ARRAY_TASK_ID"
fi
if [[ "$USER_SET_NUM_SHARDS" -eq 0 ]]; then
  if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
    NUM_SHARDS="$SLURM_ARRAY_TASK_COUNT"
  elif [[ -n "${SLURM_ARRAY_TASK_MIN:-}" && -n "${SLURM_ARRAY_TASK_MAX:-}" ]]; then
    NUM_SHARDS="$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))"
  fi
fi

if ! [[ "$NUM_SHARDS" =~ ^[0-9]+$ ]] || ! [[ "$SHARD_INDEX" =~ ^[0-9]+$ ]]; then
  echo "[x] --num-shards and --shard-index must be non-negative integers"
  exit 1
fi
if (( NUM_SHARDS < 1 )); then
  echo "[x] --num-shards must be >= 1"
  exit 1
fi
if (( SHARD_INDEX >= NUM_SHARDS )); then
  echo "[x] --shard-index ($SHARD_INDEX) must be < --num-shards ($NUM_SHARDS)"
  exit 1
fi

# Fixed round-robin sharding: assign task_idx where task_idx % NUM_SHARDS == SHARD_INDEX.
SELECTED_TASKS=()
for idx in "${!TASKS[@]}"; do
  if (( idx % NUM_SHARDS == SHARD_INDEX )); then
    SELECTED_TASKS+=("${TASKS[$idx]}")
  fi
done

if [[ ${#SELECTED_TASKS[@]} -eq 0 ]]; then
  echo "[x] No tasks selected for shard $SHARD_INDEX/$NUM_SHARDS"
  exit 1
fi

declare -A TASK_HORIZONS=()
if (( USE_TASK_HORIZON_INT == 1 )) && [[ -f "$TASK_YAML" ]]; then
  while IFS=, read -r task_name task_horizon; do
    if [[ -n "$task_name" && "$task_horizon" =~ ^[0-9]+$ ]]; then
      TASK_HORIZONS["$task_name"]="$task_horizon"
    fi
  done < <(load_task_horizons_from_yaml "$TASK_YAML")
  if (( ${#TASK_HORIZONS[@]} > 0 )); then
    echo "[i] Task horizon auto mode: enabled (${#TASK_HORIZONS[@]} tasks loaded from task yaml)"
  else
    echo "[i] Task horizon auto mode: enabled but no task_horizons found in task yaml; using fallback max_episode_steps=$MAX_EPISODE_STEPS"
  fi
else
  echo "[i] Task horizon auto mode: disabled (using --max-episode-steps=$MAX_EPISODE_STEPS)"
fi

RUN_ID="${SLURM_ARRAY_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="$(basename "$MODEL_PATH")_${TASK_SET}_${SPLIT}_exp${RUN_ID}"
EXP_DIR="$OUTPUT_ROOT/$EXP_NAME"
RUN_DIR="$EXP_DIR"
SHARD_TAG="shard${SHARD_INDEX}of${NUM_SHARDS}"
SERVER_LOG_PATH="$RUN_DIR/server_${SHARD_TAG}.log"
mkdir -p "$RUN_DIR"

printf "[i] Loaded %d tasks from set '%s'\n" "${#TASKS[@]}" "$TASK_SET"
printf "[i] Shard config: index=%d num_shards=%d selected=%d\n" "$SHARD_INDEX" "$NUM_SHARDS" "${#SELECTED_TASKS[@]}"
echo "[i] Experiment dir: $EXP_DIR"
echo "[i] Output: $RUN_DIR"
if (( WANDB_ENABLED_INT == 1 )); then
  echo "[i] W&B enabled: project=$WANDB_PROJECT run_id=${WANDB_RUN_ID:-auto} run_name=${WANDB_RUN_NAME:-auto}"
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[i] Stopping server pid=$SERVER_PID"
    kill "$SERVER_PID" || true
  fi
}
trap cleanup EXIT

uv run python "$PROJECT_REPO/rldx/eval/run_rldx_server.py" \
  --model-path "$MODEL_PATH" \
  --embodiment-tag "$EMBODIMENT_TAG" \
  --use-sim-policy-wrapper \
  --host "$SERVER_BIND_HOST" \
  --port "$SERVER_PORT" \
  --device "$SERVER_DEVICE" \
  > "$SERVER_LOG_PATH" 2>&1 &
SERVER_PID=$!

echo "[i] Server started (pid=$SERVER_PID), warming up ${SERVER_WARMUP_SEC}s..."
sleep "$SERVER_WARMUP_SEC"

SUMMARY_CSV="$RUN_DIR/summary_${SHARD_TAG}.csv"
echo "task,success_rate,log_file" > "$SUMMARY_CSV"
WANDB_STEP_BASE=$((SHARD_INDEX * 100000))
TASK_COUNTER=0

for task in "${SELECTED_TASKS[@]}"; do
  env_name="robocasa/${task}"
  task_dir="$RUN_DIR/$task"
  mkdir -p "$task_dir"
  log_file="$task_dir/eval.log"
  task_max_episode_steps="$MAX_EPISODE_STEPS"
  if (( USE_TASK_HORIZON_INT == 1 )) && [[ -n "${TASK_HORIZONS[$task]:-}" ]]; then
    task_max_episode_steps="${TASK_HORIZONS[$task]}"
  fi

  echo "[i] Evaluating $env_name (max_episode_steps=$task_max_episode_steps) ..."
  "$PY365" "$PROJECT_REPO/rldx/eval/rollout_policy.py" \
    --n_episodes "$N_EPISODES" \
    --policy_client_host "$SERVER_HOST" \
    --policy_client_port "$SERVER_PORT" \
    --max_episode_steps "$task_max_episode_steps" \
    --env_name "$env_name" \
    --n_action_steps "$N_ACTION_STEPS" \
    --n_envs "$N_ENVS" \
    --robocasa_split "$SPLIT" \
    --video_dir "$task_dir/videos" \
    > "$log_file" 2>&1

  rate="$("$PY365" - <<PY
import re
p = r"$log_file"
rate = "NA"
with open(p, "r", encoding="utf-8") as f:
    for line in f:
        m = re.search(r"success rate:\s*([0-9.]+)", line)
        if m:
            rate = m.group(1)
print(rate)
PY
)"
  echo "${task},${rate},${log_file}" >> "$SUMMARY_CSV"

  if [[ "$rate" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    metric_key="eval/${SPLIT}/${TASK_SET}/${task}/success_rate"
    metrics_json="$(python3 - <<PY
import json
print(json.dumps({
  "$metric_key": float("$rate"),
  "eval/$SPLIT/$TASK_SET/shard_index": int("$SHARD_INDEX"),
  "eval/$SPLIT/$TASK_SET/num_shards": int("$NUM_SHARDS"),
  "eval/$SPLIT/$TASK_SET/num_episodes": int("$N_EPISODES"),
}))
PY
)"
    wandb_log_metrics_json "$metrics_json" "$((WANDB_STEP_BASE + TASK_COUNTER))"
  fi
  TASK_COUNTER=$((TASK_COUNTER + 1))
done

SHARD_MEAN="$("$PY365" - <<PY
import csv
rates = []
with open("$SUMMARY_CSV", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            rates.append(float(row["success_rate"]))
        except Exception:
            pass
if rates:
    print(sum(rates) / len(rates))
else:
    print("nan")
PY
)"

if [[ "$SHARD_MEAN" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  shard_metrics_json="$(python3 - <<PY
import json
print(json.dumps({
  "eval/$SPLIT/$TASK_SET/shard_mean_success_rate": float("$SHARD_MEAN"),
  "eval/$SPLIT/$TASK_SET/shard_index": int("$SHARD_INDEX"),
}))
PY
)"
  wandb_log_metrics_json "$shard_metrics_json" "$((WANDB_STEP_BASE + 99999))"
fi

SERVER_LOG="$SERVER_LOG_PATH"
ARTIFACT_FILES=("$SUMMARY_CSV")
if [[ -f "$SERVER_LOG" ]]; then
  ARTIFACT_FILES+=("$SERVER_LOG")
fi
wandb_log_artifact "robocasa365_eval_shard_${SHARD_INDEX}_tables" "eval-shard" "${ARTIFACT_FILES[@]}"

VIDEO_ARTIFACT_FILES=()
if (( WANDB_LOG_VIDEOS_INT == 1 )); then
  for task in "${SELECTED_TASKS[@]}"; do
    task_video_count=0
    for video_file in "$RUN_DIR/$task/videos"/*.mp4; do
      if [[ ! -f "$video_file" ]]; then
        continue
      fi
      bytes="$(stat -c%s "$video_file" 2>/dev/null || echo 0)"
      max_bytes="$((WANDB_MAX_VIDEO_MB * 1024 * 1024))"
      if (( bytes <= max_bytes )); then
        VIDEO_ARTIFACT_FILES+=("$video_file")
        task_video_count=$((task_video_count + 1))
      fi
      if (( task_video_count >= WANDB_MAX_VIDEOS_PER_TASK )); then
        break
      fi
    done
  done
fi
if [[ "${#VIDEO_ARTIFACT_FILES[@]}" -gt 0 ]]; then
  wandb_log_artifact "robocasa365_eval_shard_${SHARD_INDEX}_videos" "eval-videos" "${VIDEO_ARTIFACT_FILES[@]}"
fi

echo "[i] Finished. Summary: $SUMMARY_CSV"
echo "[i] Shard mean success rate: $SHARD_MEAN"
echo "[i] To log merged summary to same W&B run after all shards finish:"
echo "    WANDB_PROJECT=\"$WANDB_PROJECT\" WANDB_ENTITY=\"$WANDB_ENTITY\" WANDB_RUN_ID=\"${WANDB_RUN_ID:-}\" WANDB_RUN_NAME=\"${WANDB_RUN_NAME:-}\" \\"
echo "    uv run --with wandb python run_scripts/eval/robocasa_365/upload_merged_eval_to_wandb.py --exp-dir \"$EXP_DIR\" --split \"$SPLIT\" --task-set \"$TASK_SET\""
