#!/bin/bash
#SBATCH --job-name="rldx-resize-dataset"
#SBATCH --partition=${SBATCH_PARTITION:-gpu}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm_out/%j-resize_dataset_videos.out
#SBATCH --error=slurm_out/%j-resize_dataset_videos.err

set -euo pipefail

# Required: DATASET_PATH — directory of LeRobot-format episodes to resize.
#
# Usage:
#   DATASET_PATH=/path/to/dataset bash run_scripts/data/resize_dataset_videos.sh
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to the LeRobot dataset directory to resize}"

BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$BASE_DIR"
mkdir -p slurm_out

uv run python run_scripts/data/resize_dataset_videos.py \
    --dataset-path "$DATASET_PATH" \
    --num-workers 32
