#!/bin/bash
#SBATCH --job-name="convert_to_lerobot"
#SBATCH --partition=${SBATCH_PARTITION:-cpu}
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-17
#SBATCH --output=slurm_out/%A_%a-convert_teleop_to_lerobot.out
#SBATCH --error=slurm_out/%A_%a-convert_teleop_to_lerobot.err

BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
SRC_DIR="$BASE_DIR/data/droid_dataset/rlds"
DST_DIR="$BASE_DIR/data/droid_dataset/lerobot"
TASK_MAP="$BASE_DIR/run_scripts/data/task_map.json"
PHYSICS_MAP="$BASE_DIR/run_scripts/data/physics_map.json"

DATASETS=(
  dust_clean_up
  dust_peg_insert
  fineact_doll_move
  franka_generalization
  hamlet_coverNstack_v2
  hamlet_cover_pickNplace_twice_stack
  hamlet_pickNplace_three_times
  hamlet_pickNplace_twice
  hamlet_swap_cubes
  pick_original
  pnp_original
  pnp_twice_original
  rldx_old_check_n_move
  rldx_old_pick_n_place_twice
  rldx_old_swap_cubes
  rldx_swap_cubes
  rscl_pnp
  tactile_moss
)

TASK_ID=${DATASETS[$SLURM_ARRAY_TASK_ID]}
echo "=== Array task $SLURM_ARRAY_TASK_ID: Converting $TASK_ID ==="

cd "$BASE_DIR"
uv run python "$BASE_DIR/run_scripts/data/convert_h5_to_v2.py" \
  --raw-dir $SRC_DIR/$TASK_ID \
  --output-path $DST_DIR/$TASK_ID \
  --repo-id $TASK_ID \
  --task-map $TASK_MAP \
  --physics-map $PHYSICS_MAP \
  --robot-type franka_panda \
  --mode video \
  --action-type eef \
  --fps 10
