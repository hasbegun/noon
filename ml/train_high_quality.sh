#!/bin/bash
# High-Quality Training Script - Option B (Target: 93-95% accuracy)
#
# This script implements the optimal settings for achieving 93-95% accuracy
# on Food-101 classification task. It uses:
# - EfficientNet-B3 backbone (larger model)
# - Mixup + CutMix augmentation
# - 150 epochs with warmup
# - Optimized batch size for MPS
#
# Expected training time: ~25-30 hours
# Expected accuracy: 93-95%

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
IMAGE_SIZE=300          # Optimal for EfficientNet-B3
BATCH_SIZE=16           # Balanced for MPS memory
EPOCHS=150              # More epochs for convergence
WARMUP_EPOCHS=5         # Warmup for stability
LEARNING_RATE=0.001     # Standard learning rate
NUM_WORKERS=4

# MPS memory management
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_MPS_ALLOCATOR_POLICY=garbage_collection

echo "============================================================"
echo "HIGH-QUALITY TRAINING - Option B"
echo "============================================================"
echo "Target: 93-95% accuracy on Food-101"
echo ""
echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Backbone: $BACKBONE"
echo "  Image size: $IMAGE_SIZE"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Warmup: $WARMUP_EPOCHS epochs"
echo "  Augmentation: Mixup + CutMix"
echo "  Device: MPS"
echo ""
echo "Estimated time: 25-30 hours"
echo "Expected accuracy: 93-95%"
echo "============================================================"
echo ""

# Clear MPS cache before starting
python -c "import torch; torch.mps.empty_cache()" 2>/dev/null || true

# Run training
python src/train/train_recognition.py \
    --dataset $DATASET \
    --backbone $BACKBONE \
    --image-size $IMAGE_SIZE \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --warmup-epochs $WARMUP_EPOCHS \
    --lr $LEARNING_RATE \
    --num-workers $NUM_WORKERS \
    --mixup \
    --cutmix \
    --mixup-alpha 0.2 \
    --cutmix-alpha 1.0 \
    --device mps \
    "$@"

echo ""
echo "============================================================"
echo "Training complete!"
echo "Check models/recognition/food-101_efficientnet_b3/ for results"
echo "============================================================"
