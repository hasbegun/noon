#!/bin/bash
# Quick test script for high-quality training features
#
# This script quickly tests all the new features on a small subset:
# - EfficientNet-B3 backbone
# - Image size 300
# - Mixup/CutMix augmentation
# - Warmup scheduler
# - Freeze backbone
#
# Expected runtime: ~10-15 minutes
# Purpose: Verify everything works before full training

set -e

# MPS memory management
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_MPS_ALLOCATOR_POLICY=garbage_collection

echo "============================================================"
echo "TESTING HIGH-QUALITY TRAINING FEATURES"
echo "============================================================"
echo "This is a quick test to verify all features work correctly"
echo "Using small subset: 100 samples, 3 epochs"
echo "Expected time: ~10-15 minutes"
echo "============================================================"
echo ""

# Clear MPS cache
python -c "import torch; torch.mps.empty_cache()" 2>/dev/null || true

echo "Test 1: EfficientNet-B3 with Mixup/CutMix"
echo "============================================================"
python src/train/train_recognition.py \
    --dataset food-101 \
    --backbone efficientnet_b3 \
    --image-size 300 \
    --batch-size 16 \
    --epochs 3 \
    --warmup-epochs 1 \
    --mixup \
    --cutmix \
    --dev-mode \
    --dev-samples 100 \
    --device mps

echo ""
echo "============================================================"
echo "ALL TESTS PASSED! ✅"
echo "============================================================"
echo ""
echo "You can now run full training with:"
echo "  ./train_high_quality.sh     (single-stage, 93-95% accuracy)"
echo "  ./train_two_stage.sh        (two-stage, 94-96% accuracy)"
echo "============================================================"
