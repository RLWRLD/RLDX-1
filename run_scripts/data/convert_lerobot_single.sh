#!/bin/bash
#SBATCH --job-name="Convert raw dataset to LeRobot format"
#SBATCH --partition=${SBATCH_PARTITION:-cpu}
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_out/%j-convert_teleop_to_lerobot.out
#SBATCH --error=slurm_out/%j-convert_teleop_to_lerobot.err

BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
SRC_DIR="$BASE_DIR/data/droid_dataset/rlds"
DST_DIR="$BASE_DIR/data/droid_dataset/lerobot"
TASK_ID="swap_cubes"
TASK_INSTRUCTION="Swap the positions of two cubes, starting with the blue one."

cd "$BASE_DIR"
uv run python "$BASE_DIR/run_scripts/data/convert_h5_to_v2.py" \
  --raw-dir $SRC_DIR/$TASK_ID \
  --output-path $DST_DIR/$TASK_ID \
  --repo-id $TASK_ID \
  --task $TASK_INSTRUCTION \
  --robot-type franka_panda \
  --mode video \
  --action-type eef \
  --fps 10