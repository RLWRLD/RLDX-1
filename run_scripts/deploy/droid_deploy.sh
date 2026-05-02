#!/bin/bash
set -euo pipefail
BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

# =========================================
# Required: set HF_MODEL_ID to the policy checkpoint you want to deploy.
# MODEL_NAME is a human-readable slug used for local logging paths.
# Override via environment:
#   HF_MODEL_ID=org/my-droid-ckpt MODEL_NAME=my-run bash droid_deploy.sh
# =========================================
HF_MODEL_ID="${HF_MODEL_ID:?Set HF_MODEL_ID to your policy checkpoint HF repo id or local path}"
MODEL_NAME="${MODEL_NAME:-$(basename "$HF_MODEL_ID")}"
LOGGING_PATH="${LOGGING_PATH:-$BASE_DIR/output/deploy/$MODEL_NAME}"
# =========================================

# =========================================
INSTRUCTION="Swap the positions of two cubes, starting with the blue one."
# "Pick up the cube and place it on the opposite side twice."
# "Lift the center cup to check the cube's color, then place the center object on the cup that matches the cube."
# "Swap the positions of two cubes, starting with the blue one."
# "Move left when the circle in the video moves left, and move right when it moves right."
# "Move right when the circle in the video moves clockwise, and move left when it moves counterclockwise."
# =========================================

"$BASE_DIR/.venv/bin/python" -u \
    "$BASE_DIR/run_scripts/deploy/droid_deploy.py" \
    --instruction="$INSTRUCTION" \
    --logging_path="$LOGGING_PATH" \
    --binarize_gripper \
    --max_timesteps=700 \
    --open_loop_horizon=16 \
    --embodiment_tag="GENERAL_EMBODIMENT" \
    --video_key_exterior="exterior_image_1_left" \
    --video_key_wrist="wrist_image_left" \
    --resize_video="168,336" \
    --policy.config="msat" \
    --policy.dir="$HF_MODEL_ID"