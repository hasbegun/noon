#!/bin/bash
# Ultra-safe training script for MPS with aggressive memory management

# Remove MPS memory limit (allows using system memory as swap)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Reduce PyTorch memory allocator fragmentation
export PYTORCH_MPS_ALLOCATOR_POLICY=garbage_collection

# Use very small batch size
BATCH_SIZE=4  # Ultra conservative

# Reduce number of workers to save memory
NUM_WORKERS=2

echo "=========================================="
echo "Ultra-Safe Training Configuration"
echo "=========================================="
echo "Batch Size: $BATCH_SIZE (ultra conservative)"
echo "Image Size: 224"
echo "Workers: $NUM_WORKERS"
echo "Device: MPS with unlimited memory"
echo "Memory Management: Aggressive"
echo "=========================================="
echo ""

# Clear any existing MPS cache
python -c "import torch; torch.mps.empty_cache()" 2>/dev/null || true

# Run training with ultra-safe settings
python src/train/train_recognition.py \
    --dataset food-101 \
    --backbone efficientnet_b0 \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --device mps \
    "$@"
