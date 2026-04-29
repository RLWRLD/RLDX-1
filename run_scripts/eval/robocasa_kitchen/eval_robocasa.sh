#!/bin/bash
# Parallel robocasa eval on 4 GPUs (local, no SLURM).
# Usage: eval_robocasa.sh <CKPT_NAME>
set -u
export NO_ALBUMENTATIONS_UPDATE=1

CKPT_NAME=${1:?"Usage: $0 <CKPT_NAME>"}
BASE_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
MODEL_PATH="$CKPT_NAME"

OUT_ROOT="$BASE_DIR/output_final/robocasa/$CKPT_NAME"
LOG_ROOT="$BASE_DIR/output_final/robocasa/$CKPT_NAME/_launcher_logs"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

TASK_NAMES=(
  "TurnSinkSpout"  "TurnOnStove"  "TurnOnSinkFaucet"  "TurnOnMicrowave"
  "TurnOffStove"  "TurnOffSinkFaucet"
  "TurnOffMicrowave"  "PnPStoveToCounter"  "PnPSinkToCounter"  "PnPMicrowaveToCounter"
  "PnPCounterToStove"  "PnPCounterToSink"
  "PnPCounterToMicrowave"  "PnPCounterToCab"  "PnPCabToCounter"  "OpenSingleDoor"
  "OpenDrawer"  "OpenDoubleDoor"
  "CoffeeSetupMug"  "CoffeeServeMug"  "CoffeePressButton"  "CloseSingleDoor"
  "CloseDrawer"  "CloseDoubleDoor"
)
N_GPUS=4
TASKS_PER_GPU=6
BASE_PORT=20100

run_shard() {
  local gpu_id=$1
  local port=$((BASE_PORT + gpu_id))
  local start=$((gpu_id * TASKS_PER_GPU))
  local end=$((start + TASKS_PER_GPU))
  local shard_log="$LOG_ROOT/shard-gpu${gpu_id}.log"

  echo "[shard ${gpu_id}] GPU=${gpu_id} PORT=${port} tasks=${start}..$((end-1))" | tee -a "$shard_log"

  CUDA_VISIBLE_DEVICES=$gpu_id uv run python "$BASE_DIR/rldx/eval/run_rldx_server.py" \
    --model-path "$MODEL_PATH" \
    --embodiment-tag GENERAL_EMBODIMENT \
    --use-sim-policy-wrapper \
    --host 127.0.0.1 \
    --port "$port" >> "$shard_log" 2>&1 &
  local serve_pid=$!
  echo "[shard ${gpu_id}] server PID=${serve_pid}" >> "$shard_log"

  # Wait for server to be ready (poll port)
  for _ in $(seq 1 120); do
    if ss -lnt | awk '{print $4}' | grep -q ":$port$"; then
      echo "[shard ${gpu_id}] server ready on port ${port}" >> "$shard_log"
      break
    fi
    sleep 2
  done

  for i in $(seq $start $((end - 1))); do
    local task_name=${TASK_NAMES[$i]}
    local out_dir="$OUT_ROOT/$task_name"
    mkdir -p "$out_dir"
    echo "[shard ${gpu_id}] running ${task_name}" >> "$shard_log"
    "$BASE_DIR/rldx/eval/sim/robocasa/robocasa_uv/.venv/bin/python" \
      "$BASE_DIR/rldx/eval/rollout_policy.py" \
        --n_episodes 50 \
        --policy_client_host 127.0.0.1 \
        --policy_client_port "$port" \
        --max_episode_steps 720 \
        --env_name "robocasa_panda_omron/${task_name}_PandaOmron_Env" \
        --n_action_steps 16 \
        --n_envs 1 \
        --video_dir "$out_dir" \
        >> "$out_dir/eval.log" 2>&1
    echo "[shard ${gpu_id}] ${task_name} done (exit=$?)" >> "$shard_log"
  done

  kill "$serve_pid" 2>/dev/null || true
  echo "[shard ${gpu_id}] shard complete" >> "$shard_log"
}

SHARD_PIDS=()
for gpu_id in $(seq 0 $((N_GPUS - 1))); do
  run_shard $gpu_id &
  SHARD_PIDS+=($!)
done

for pid in "${SHARD_PIDS[@]}"; do
  wait "$pid"
done

echo "[launcher] All shards complete"
