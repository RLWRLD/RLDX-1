#!/bin/bash
export WANDB_PROJECT="${WANDB_PROJECT:-rldx-finetune}"
export NO_ALBUMENTATIONS_UPDATE=1
# =========================================
BASE_MODEL_PATH="${BASE_MODEL_PATH:-RLWRLD/RLDX-1-PT}"
CKPT_NAME="rldx1_ft_robocasa300_bsz1024_lr4e4_60k"
RUN_NAME="rldx1_ft_robocasa300_bsz1024_lr4e4_60k"
# =========================================

NUM_GPUS=8
BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the robocasa_mg_gr00t_300 dataset root}"

CKPT_DIR="$BASE_DIR/ckpt/rldx1/finetuned/$CKPT_NAME"
MODALITY_CONFIG_PATH="$BASE_DIR/rldx/configs/data/robocasa_config.py"
COLOR_JITTER_PARAMS="brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08"

# ── Component configs ─────────────────────────────────────
VIDEO_CONFIGS="\
    --video-length 4"
# ──────────────────────────────────────────────────────────

# 의존성 설치
apt-get update && apt-get install -y ffmpeg iproute2 libibverbs-dev ibverbs-utils -q
cd $BASE_DIR
uv sync
uv pip install -e .

cd $BASE_DIR
export MASTER_PORT=$(shuf -i 20000-30000 -n 1)
uv run torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    rldx/experiment/launch_train.py \
        --n-cog-tokens 64 \
        $VIDEO_CONFIGS \
        --dataset-path $DATA_DIR \
        --dataloader-num-workers 12 \
        --embodiment-tag GENERAL_EMBODIMENT \
        --modality-config-path $MODALITY_CONFIG_PATH \
        --color-jitter-params $COLOR_JITTER_PARAMS \
        --base-model-path $BASE_MODEL_PATH \
        --output-dir $CKPT_DIR \
        --num-gpus $NUM_GPUS \
        --save-total-limit 5 \
        --save-steps 1000 \
        --max-steps 60000 \
        --learning-rate 4e-4 \
        --global-batch-size 512 \
        --gradient-accumulation-steps 2 \
        --use-wandb \
        --wandb-project $WANDB_PROJECT \
        --experiment-name $RUN_NAME
