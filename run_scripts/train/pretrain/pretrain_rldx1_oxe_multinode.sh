#!/bin/bash
#SBATCH --job-name="Pretrain RLDX-1 (video-only) on OXE magic_soup_plus"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --partition=${SBATCH_PARTITION:-gpu}
#SBATCH --output=slurm_out/%j-rldx1_pt_oxeMagicSoupPlus_bsz1024_100k.out
#SBATCH --error=slurm_out/%j-rldx1_pt_oxeMagicSoupPlus_bsz1024_100k.err

export WANDB_PROJECT="${WANDB_PROJECT:-rldx-pretrain}"
export NO_ALBUMENTATIONS_UPDATE=1

# =========================================
BACKBONE_PATH="TBD_BACKBONE_PATH"  # TODO(release): e.g. Qwen/Qwen3-VL-8B-Instruct  # "Qwen/Qwen3-VL-8B-Instruct"
CKPT_NAME="rldx1_pt_oxeMagicSoupPlus_bsz1024_100k"
RUN_NAME="rldx1_pt_oxeMagicSoupPlus_bsz1024_100k"
# =========================================

NUM_GPUS=8
NUM_NODES=2
TOTAL_GPUS=$(( NUM_GPUS * NUM_NODES ))
BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the pretraining dataset root}"
DATA_MIX="oxe_magic_soup_plus"

CKPT_DIR="$BASE_DIR/ckpt/rldx1/pretrained/$CKPT_NAME"
MODALITY_CONFIG_PATH="$BASE_DIR/rldx/configs/data/pt_data_config.py"
COLOR_JITTER_PARAMS="brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08"

# ── Component configs ─────────────────────────────────────
VIDEO_CONFIGS="\
    --video-length 4"
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
        --pt-dataset-root $DATA_ROOT \
        --pt-dataset-mix $DATA_MIX \
        --dataloader-num-workers 40 \
        --modality-config-path $MODALITY_CONFIG_PATH \
        --color-jitter-params $COLOR_JITTER_PARAMS \
        --backbone-path $BACKBONE_PATH \
        --output-dir $CKPT_DIR \
        --num-gpus $TOTAL_GPUS \
        --save-total-limit 40 \
        --save-steps 1000 \
        --max-steps 100000 \
        --global-batch-size 512 \
        --gradient-accumulation-steps 2 \
        --lr-scheduler-type constant_with_warmup \
        --use-wandb \
        --wandb-project $WANDB_PROJECT \
        --experiment-name $RUN_NAME
