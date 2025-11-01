#!/bin/bash
# Two-Stage High-Quality Training Script
#
# Stage 1: Train classifier with frozen backbone (10 epochs)
# Stage 2: Fine-tune entire model (100 epochs) with mixup/cutmix
#
# This approach often achieves better results than single-stage training
# Expected accuracy: 94-96%
# Total time: ~30-35 hours

set -e  # Exit on error

# Increase file descriptor limit (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS - Setting file descriptor limits"

    # Soft limit
    ulimit -n 4096

    # Show current limits
    echo "File descriptor limits:"
    echo "  Soft limit: $(ulimit -n)"
    echo "  Hard limit: $(ulimit -Hn)"
    echo ""
fi

# Set PyTorch multiprocessing strategy
export PYTORCH_MULTIPROCESSING_STRATEGY=file_system

# Configuration
DATASET="food-101"
BACKBONE="efficientnet_b3"
IMAGE_SIZE=300
BATCH_SIZE=16
NUM_WORKERS=4

# MPS memory management
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_MPS_ALLOCATOR_POLICY=garbage_collection

echo "============================================================"
echo "TWO-STAGE HIGH-QUALITY TRAINING"
echo "============================================================"
echo "Stage 1: Freeze backbone, train classifier (10 epochs)"
echo "Stage 2: Fine-tune entire model (100 epochs + augmentation)"
echo ""
echo "Target: 94-96% accuracy"
echo "Total time: ~30-35 hours"
echo "============================================================"
echo ""

# Clear MPS cache
python -c "import torch; torch.mps.empty_cache()" 2>/dev/null || true

#############################################
# STAGE 1: Train classifier with frozen backbone
#############################################
echo ""
echo "============================================================"
echo "STAGE 1: Training classifier with frozen backbone"
echo "============================================================"
echo ""

python src/train/train_recognition.py \
    --dataset $DATASET \
    --backbone $BACKBONE \
    --image-size $IMAGE_SIZE \
    --batch-size $BATCH_SIZE \
    --epochs 10 \
    --lr 0.001 \
    --num-workers $NUM_WORKERS \
    --freeze-backbone \
    --device mps \
    "$@"

echo ""
echo "Stage 1 complete! Proceeding to Stage 2..."
echo ""
sleep 3

#############################################
# STAGE 2: Fine-tune entire model
#############################################
echo ""
echo "============================================================"
echo "STAGE 2: Fine-tuning entire model with augmentation"
echo "============================================================"
echo ""

# Get the last checkpoint from stage 1
CHECKPOINT_DIR="models/recognition/${DATASET}_${BACKBONE}"
CHECKPOINT="${CHECKPOINT_DIR}/last_checkpoint.pt"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    echo "Stage 1 may have failed. Please check logs."
    exit 1
fi

python src/train/train_recognition.py \
    --dataset $DATASET \
    --backbone $BACKBONE \
    --image-size $IMAGE_SIZE \
    --batch-size $BATCH_SIZE \
    --epochs 100 \
    --warmup-epochs 5 \
    --lr 0.0001 \
    --num-workers $NUM_WORKERS \
    --mixup \
    --cutmix \
    --mixup-alpha 0.2 \
    --cutmix-alpha 1.0 \
    --checkpoint $CHECKPOINT \
    --device mps \
    "$@"

echo ""
echo "============================================================"
echo "TWO-STAGE TRAINING COMPLETE!"
echo "============================================================"
echo "Best models saved in: $CHECKPOINT_DIR"
echo "  - best_accuracy.pt: Best validation accuracy"
echo "  - best_f1.pt: Best F1 score"
echo "  - last_checkpoint.pt: Final model"
echo ""
echo "Expected accuracy: 94-96%"
echo "============================================================"
