#!/usr/bin/env bash
# Local GR-1 Tabletop evaluation runner.
#
# Usage:
#   bash run_scripts/eval/gr1_tabletop/eval_gr1.sh [MODEL_PATH]
#
# Default MODEL_PATH is RLWRLD/RLDX-1-FT-GR1.

set -euo pipefail
export NO_ALBUMENTATIONS_UPDATE=1
export IS_RLDX_EVAL=1

MODEL_PATH="${1:-RLWRLD/RLDX-1-FT-GR1}"
BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
GR1_PY="$BASE_DIR/rldx/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python"
PORT="${PORT:-20100}"
N_EPISODES="${N_EPISODES:-50}"

TASKS=(
  "PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env"
  "PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"
  "PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"
  "PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env"
  "PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env"
  "PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env"
  "PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env"
)

TAG=$(echo "$MODEL_PATH" | tr '/' '_')
OUT_ROOT="$BASE_DIR/output_final/gr1_tabletop/$TAG"
mkdir -p "$OUT_ROOT"

echo "[i] Model path : $MODEL_PATH"
echo "[i] Tasks      : ${#TASKS[@]}"
echo "[i] Episodes   : $N_EPISODES per task"

cd "$BASE_DIR"
uv run python rldx/eval/run_rldx_server.py \
    --model-path "$MODEL_PATH" \
    --embodiment-tag GR1 \
    --use-sim-policy-wrapper \
    --host 127.0.0.1 --port "$PORT" &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null || true' EXIT

sleep 30

for task in "${TASKS[@]}"; do
  out_dir="$OUT_ROOT/$task"
  mkdir -p "$out_dir"
  echo "[i] === $task ==="
  "$GR1_PY" "$BASE_DIR/rldx/eval/rollout_policy.py" \
      --policy_client_host 127.0.0.1 --policy_client_port "$PORT" \
      --env_name "gr1_unified/$task" \
      --n_episodes "$N_EPISODES" \
      --max_episode_steps 720 \
      --n_action_steps 16 \
      --n_envs 1 \
      --video_dir "$out_dir" 2>&1 | tee "$out_dir/eval.log"
done

echo "[i] All done. Outputs under $OUT_ROOT"
