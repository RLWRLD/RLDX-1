#!/usr/bin/env bash
# Local LIBERO-Plus evaluation runner.
#
# Usage:
#   bash run_scripts/eval/libero_plus/eval_libero_plus.sh [MODEL_PATH] [SUITE_FILTER]
#
# Default MODEL_PATH is RLWRLD/RLDX-1-FT-LIBERO. Optional SUITE_FILTER restricts
# evaluation to a single LIBERO suite (libero_10 / libero_spatial /
# libero_object / libero_goal).
#
# `LIBERO_PLUS_DATA_DIR` must point at the LIBERO-Plus dataset checkout.

set -euo pipefail
export NO_ALBUMENTATIONS_UPDATE=1

MODEL_PATH="${1:-RLWRLD/RLDX-1-FT-LIBERO}"
SUITE_FILTER="${2:-}"

BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
LIBERO_PLUS_VENV="$BASE_DIR/rldx/eval/sim/LIBERO_PLUS/libero_plus_uv/.venv"
LIBERO_PLUS_REPO="${LIBERO_PLUS_DATA_DIR:?Set LIBERO_PLUS_DATA_DIR to the LIBERO-Plus dataset checkout}"
export LIBERO_CONFIG_PATH="$LIBERO_PLUS_REPO/.libero_config"

CKPT_NAME=$(basename "$MODEL_PATH")
PORT="${PORT:-20201}"
N_EPISODES="${N_EPISODES:-1}"   # LIBERO-Plus paper: 1 trial per perturbed task
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"

if [ ! -f "$LIBERO_PLUS_VENV/bin/python" ]; then
  echo "[ERROR] LIBERO-Plus venv not found. Run: bash run_scripts/eval/libero_plus/setup_libero_plus.sh" >&2
  exit 1
fi

echo "[i] Model        : $MODEL_PATH"
echo "[i] Suite filter : ${SUITE_FILTER:-all}"
echo "[i] Port         : $PORT"

cd "$BASE_DIR"
uv run python rldx/eval/run_rldx_server.py \
    --model-path "$MODEL_PATH" \
    --embodiment-tag GENERAL_EMBODIMENT \
    --use-sim-policy-wrapper \
    --host 127.0.0.1 --port "$PORT" &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null || true' EXIT

for i in $(seq 1 90); do
  ss -lnt | grep -q ":$PORT " && break
  sleep 2
done

# Build full task list across (filtered) suites
TASK_LIST=$("$LIBERO_PLUS_VENV/bin/python" - "$SUITE_FILTER" <<'PYEOF'
import contextlib, io, sys
from libero.libero import benchmark

suite_filter = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None
suites = ["libero_10", "libero_spatial", "libero_object", "libero_goal"]
if suite_filter:
    suites = [s for s in suites if s == suite_filter]

with contextlib.redirect_stdout(io.StringIO()):
    bd = benchmark.get_benchmark_dict()
    suite_objs = {s: bd[s]() for s in suites if s in bd}

for name, suite in suite_objs.items():
    for task_id in range(suite.get_num_tasks()):
        print(f"{name}\t{suite.get_task(task_id).name}")
PYEOF
)

IFS=$'\n' read -r -d '' -a TASK_LINES <<< "$TASK_LIST" || true
echo "[i] Total tasks  : ${#TASK_LINES[@]}"

OUT_ROOT="$BASE_DIR/output_final/libero_plus/$CKPT_NAME"
for line in "${TASK_LINES[@]}"; do
  SUITE=$(echo "$line" | cut -f1)
  TASK_NAME=$(echo "$line" | cut -f2)
  out_dir="$OUT_ROOT/$SUITE/$TASK_NAME"
  mkdir -p "$out_dir"

  echo "[i] === $SUITE / $TASK_NAME ==="
  "$LIBERO_PLUS_VENV/bin/python" \
      "$BASE_DIR/rldx/eval/rollout_policy.py" \
      --n_episodes "$N_EPISODES" \
      --policy_client_host 127.0.0.1 --policy_client_port "$PORT" \
      --max_episode_steps 720 \
      --env_name "libero_plus_sim/$TASK_NAME" \
      --n_action_steps "$N_ACTION_STEPS" \
      --n_envs 1 \
      --video_dir "$out_dir" 2>&1 | tee "$out_dir/eval.log"
done

echo "[i] All done. Outputs under $OUT_ROOT"
