#!/bin/bash
# Safe training script for MPS (Apple Silicon) with memory management

# Increase file descriptor limit (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS - Setting file descriptor limits"
    ulimit -n 4096
    echo "File descriptor limit: $(ulimit -n)"
    echo ""
fi

# Set PyTorch multiprocessing strategy
export PYTORCH_MULTIPROCESSING_STRATEGY=file_system

# Set MPS memory management
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Use smaller batch size to prevent OOM
BATCH_SIZE=8  # Reduced from 32

echo "=========================================="
echo "Safe Training Configuration for MPS"
echo "=========================================="
echo "Batch Size: $BATCH_SIZE"
echo "Image Size: 224 (default)"
echo "Device: MPS with memory management"
echo "=========================================="
echo ""

# Run training with safer settings
python src/train/train_recognition.py \
    --dataset food-101 \
    --backbone efficientnet_b0 \
    --dev-mode \
    --dev-samples 100 \
    --epochs 2 \
    --batch-size $BATCH_SIZE \
    --device mps \
    "$@"
