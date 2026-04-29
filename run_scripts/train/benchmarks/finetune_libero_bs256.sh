#!/bin/bash
#SBATCH --job-name="Finetune RLDX-1 on LIBERO (bsz256)"
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --partition=${SBATCH_PARTITION:-gpu}
#SBATCH --output=slurm_out/%j-rldx1_ft_libero_bsz256_60k.out
#SBATCH --error=slurm_out/%j-rldx1_ft_libero_bsz256_60k.err

export WANDB_PROJECT="${WANDB_PROJECT:-rldx-finetune}"
export NO_ALBUMENTATIONS_UPDATE=1

# =========================================
BASE_MODEL_PATH="${BASE_MODEL_PATH:-RLWRLD/RLDX-1-PT}"
CKPT_NAME="rldx1_ft_libero_bsz256_60k"
RUN_NAME="rldx1_ft_libero_bsz256_60k"
# =========================================

NUM_GPUS=8
BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
# Override with: DATA_DIR=/your/path bash finetune_libero_bs256.sh
DATA_DIR="${DATA_DIR:-/path/to/libero_delta}"

CKPT_DIR="$BASE_DIR/ckpt/rldx1/finetuned/$CKPT_NAME"
MODALITY_CONFIG_PATH="$BASE_DIR/rldx/configs/data/libero_config.py"
COLOR_JITTER_PARAMS="brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08"

# ── Component configs ─────────────────────────────────────
VIDEO_CONFIGS="\
    --video-length 4"
# ──────────────────────────────────────────────────────────

cd $BASE_DIR
export MASTER_PORT=$(shuf -i 20000-30000 -n 1)
uv run torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    rldx/experiment/launch_train.py \
        --n-cog-tokens 64 \
        $VIDEO_CONFIGS \
        --dataset-path $DATA_DIR \
        --dataloader-num-workers 8 \
        --embodiment-tag GENERAL_EMBODIMENT \
        --modality-config-path $MODALITY_CONFIG_PATH \
        --color-jitter-params $COLOR_JITTER_PARAMS \
        --base-model-path $BASE_MODEL_PATH \
        --output-dir $CKPT_DIR \
        --num-gpus $NUM_GPUS \
        --save-total-limit 2 \
        --save-steps 5000 \
        --max-steps 60000 \
        --global-batch-size 256 \
        --use-wandb \
        --wandb-project $WANDB_PROJECT \
        --experiment-name $RUN_NAME
