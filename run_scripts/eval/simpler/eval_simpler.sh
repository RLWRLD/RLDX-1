#!/usr/bin/env bash
# Local SimplerEnv evaluation runner.
#
# Usage:
#   bash run_scripts/eval/simpler/eval_simpler.sh google_vm <model_path>
#   bash run_scripts/eval/simpler/eval_simpler.sh google_va <model_path>
#   bash run_scripts/eval/simpler/eval_simpler.sh widowx    <model_path>
#
# Default <model_path> is the matching released checkpoint:
#   google_vm / google_va -> RLWRLD/RLDX-1-FT-SIMPLER-GOOGLE
#   widowx               -> RLWRLD/RLDX-1-FT-SIMPLER-WIDOWX

set -euo pipefail
export NO_ALBUMENTATIONS_UPDATE=1

VARIANT="${1:?Usage: eval_simpler.sh <google_vm|google_va|widowx> [MODEL_PATH]}"
MODEL_PATH="${2:-}"

BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
SIMPLER_PY="$BASE_DIR/rldx/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python"
PORT=${PORT:-20100}

case "$VARIANT" in
  google_vm)
    MODEL_PATH="${MODEL_PATH:-RLWRLD/RLDX-1-FT-SIMPLER-GOOGLE}"
    EMBODIMENT_TAG="OXE_BRIDGE_ORIG"
    DENOISE_STEP=10
    ACTION_STEPS=2
    N_EPISODES=200
    TASKS=(
      "simpler_env_google/google_robot_pick_coke_can"
      "simpler_env_google/google_robot_move_near"
      "simpler_env_google/google_robot_open_drawer"
      "simpler_env_google/google_robot_close_drawer"
    )
    ;;
  google_va)
    MODEL_PATH="${MODEL_PATH:-RLWRLD/RLDX-1-FT-SIMPLER-GOOGLE}"
    EMBODIMENT_TAG="OXE_BRIDGE_ORIG"
    DENOISE_STEP=10
    ACTION_STEPS=2
    N_EPISODES=100
    BASE_TASKS=(
      "google_robot_pick_coke_can"
      "google_robot_move_near"
      "google_robot_open_drawer"
      "google_robot_close_drawer"
    )
    DRAWER_VARIANTS=(base bg_bedroom bg_office light_brighter light_darker cab_station2 cab_station3)
    PICK_MOVE_VARIANTS=(base table_cabinet1 table_cabinet2 bg_alt1 bg_alt2 light_darker light_brighter)
    TASKS=()
    for t in "${BASE_TASKS[@]}"; do
      if [[ "$t" == *drawer* ]]; then VARIANTS=("${DRAWER_VARIANTS[@]}"); else VARIANTS=("${PICK_MOVE_VARIANTS[@]}"); fi
      for v in "${VARIANTS[@]}"; do
        TASKS+=("simpler_env_google_va/${t}_${v}")
      done
    done
    ;;
  widowx)
    MODEL_PATH="${MODEL_PATH:-RLWRLD/RLDX-1-FT-SIMPLER-WIDOWX}"
    EMBODIMENT_TAG="OXE_BRIDGE_ORIG"
    DENOISE_STEP=10
    ACTION_STEPS=2
    N_EPISODES=200
    TASKS=(
      "simpler_env_widowx/widowx_spoon_on_towel"
      "simpler_env_widowx/widowx_carrot_on_plate"
      "simpler_env_widowx/widowx_put_eggplant_in_basket"
      "simpler_env_widowx/widowx_stack_cube"
    )
    ;;
  *)
    echo "[e] Unknown variant: $VARIANT (expected google_vm | google_va | widowx)" >&2
    exit 1
    ;;
esac

TAG=$(echo "$MODEL_PATH" | tr '/' '_')
OUT_ROOT="$BASE_DIR/output_final/simpler_${VARIANT}/${TAG}"
mkdir -p "$OUT_ROOT"

echo "[i] Variant     : $VARIANT"
echo "[i] Model path  : $MODEL_PATH"
echo "[i] Embodiment  : $EMBODIMENT_TAG"
echo "[i] Tasks       : ${#TASKS[@]}"

cd "$BASE_DIR"
uv run python rldx/eval/run_rldx_server.py \
    --model-path "$MODEL_PATH" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --use-sim-policy-wrapper \
    --num-inference-timesteps "$DENOISE_STEP" \
    --host 127.0.0.1 --port "$PORT" &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null || true' EXIT

sleep 30

run_eval() {
  local env_name="$1"
  local out_dir="$2"
  local cmd=("$SIMPLER_PY" "$BASE_DIR/rldx/eval/rollout_policy.py"
    --n_episodes "$N_EPISODES"
    --policy_client_host 127.0.0.1 --policy_client_port "$PORT"
    --max_episode_steps 300
    --env_name "$env_name"
    --n_action_steps "$ACTION_STEPS"
    --n_envs 5
    --video_dir "$out_dir")
  if command -v xvfb-run &>/dev/null; then
    xvfb-run -a "${cmd[@]}"
  else
    MUJOCO_GL=egl "${cmd[@]}"
  fi
}

for env_name in "${TASKS[@]}"; do
  task_label="${env_name#simpler_env_*/}"
  out_dir="$OUT_ROOT/$task_label"
  mkdir -p "$out_dir"
  echo "[i] === $env_name ==="
  run_eval "$env_name" "$out_dir" 2>&1 | tee "$out_dir/eval.log"
done

echo "[i] All done. Outputs under $OUT_ROOT"
