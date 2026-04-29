#!/bin/bash
#SBATCH --job-name="Finetune RLDX-1 (motion + memory + physics) on DROID dataset"
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --partition=${SBATCH_PARTITION:-gpu}
#SBATCH --output=slurm_out/%j-rldx1_midtrain_droid.out
#SBATCH --error=slurm_out/%j-rldx1_midtrain_droid.err

export WANDB_PROJECT="${WANDB_PROJECT:-rldx-finetune}"
export NO_ALBUMENTATIONS_UPDATE=1

# =========================================
BASE_MODEL_PATH="${BASE_MODEL_PATH:-RLWRLD/RLDX-1-PT}"
CKPT_NAME="rldx1_midtrain_droid"
RUN_NAME="rldx1_midtrain_droid"
# =========================================

NUM_GPUS=8
NUM_NODES=8
TOTAL_GPUS=$(( NUM_GPUS * NUM_NODES ))
BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
# Override with: DATA_ROOT=/your/path bash midtrain_rldx1_droid.sh
DATA_ROOT="${DATA_ROOT:-/path/to/mid_train_droid_0328_final}"
DATA_MIX="rldx1_midtrain_droid"

CKPT_DIR="$BASE_DIR/ckpt/rldx1/finetuned/$CKPT_NAME"
MODALITY_CONFIG_PATH="$BASE_DIR/rldx/configs/data/midtrain_allex_data_config.py"
COLOR_JITTER_PARAMS="brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08"

# ── Component configs ─────────────────────────────────────
VIDEO_CONFIGS="\
    --video-length 4"

MOSS_CONFIGS="\
    --use-motion \
    --motion-insert-layer 9"

MEMORY_CONFIGS="\
    --use-memory \
    --memory-length 4 \
    --memory-n-cog-tokens 16 \
    --concat-memory \
    --memory-dropout-prob 0.2"

PHYSICS_CONFIGS="\
    --use-physics \
    --physics-keys tactile torque \
    --physics-dims 30 7"

MID_TRAINING_CONFIGS="\
    --allow-missing-physics \
    --new-param-warmup-steps 2000"
# ──────────────────────────────────────────────────────────
# Multi-node rendezvous setup ============================================
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(shuf -i 20000-30000 -n 1)
export WORLD_SIZE=$(( NUM_NODES * NUM_GPUS ))
# ========================================================================

cd $BASE_DIR    
srun uv run torchrun --nproc_per_node=$NUM_GPUS --nnodes=$NUM_NODES \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    rldx/experiment/launch_train.py \
        --n-cog-tokens 64 \
        $VIDEO_CONFIGS \
        $MOSS_CONFIGS \
        $MEMORY_CONFIGS \
        $PHYSICS_CONFIGS \
        $MID_TRAINING_CONFIGS \
        --pt-dataset-root $DATA_ROOT \
        --pt-dataset-mix $DATA_MIX \
        --dataloader-num-workers 8 \
        --modality-config-path $MODALITY_CONFIG_PATH \
        --color-jitter-params $COLOR_JITTER_PARAMS \
        --base-model-path $BASE_MODEL_PATH \
        --output-dir $CKPT_DIR \
        --num-gpus $NUM_GPUS \
        --save-total-limit 40 \
        --save-steps 1000 \
        --max-steps 30_000 \
        --global-batch-size 512 \
        --gradient-accumulation-steps 4 \
        --learning-rate 5e-5 \
        --lr-scheduler-type constant_with_warmup \
        --use-wandb \
        --wandb-project $WANDB_PROJECT \
        --experiment-name $RUN_NAME


# 64 GPU, bs2048 -> 32
# 64 GPU, bs1024 -> 16 (non-available)
# 64 GPU, bs512 -> 8 (available)