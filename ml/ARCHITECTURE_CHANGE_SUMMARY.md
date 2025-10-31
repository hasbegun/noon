# Architecture Change: Segmentation → Recognition

## Summary

Changed ML training approach from **segmentation** to **food recognition (classification)** to align with available data and project goals.

## Problem

- Original approach: Train segmentation model to detect food pixels
- Dataset (Food-101): 101k images with **classification labels only**, no segmentation masks
- Result: Model trained on all-zero masks → useless output (F1=0.0, but "accuracy"=1.0)

## Solution

New 3-stage pipeline:
1. **SAM2** (pre-trained) - segments food regions
2. **Recognition Model** (trainable) - classifies detected items
3. **USDA Lookup** (existing) - provides nutrition data

## Changes Made

### New Files

**Models** (4 files):
- `src/models/food_recognition.py` - FoodRecognitionModel + FoodRecognitionWithNutrition

**Data** (2 files):
- `src/data_process/food_labels.py` - Label management (Food-101, Nutrition5k)
- `src/data_process/classification_dataset.py` - Classification dataset + data loader

**Training** (3 files):
- `src/training/classification_trainer.py` - Trainer for classification
- `src/training/classification_metrics.py` - Accuracy, Precision, Recall, F1
- `src/training/classification_losses.py` - Combined CE + Focal loss

**Scripts** (2 files):
- `src/train/train_recognition.py` - Training script
- `src/train/inference_recognition.py` - Full pipeline inference

**Documentation** (2 files):
- `RECOGNITION_ARCHITECTURE.md` - Complete architecture guide
- `ARCHITECTURE_CHANGE_SUMMARY.md` - This file

### Modified Files

**Configuration** (1 file):
- `src/config.py` - Added `recognition_models_path`

**Data Processing** (2 files):
- `src/data_process/preprocessing.py` - Extract food_class for Nutrition5k
- `src/models/__init__.py` - Export new models

## Quick Start

```bash
# 1. Preprocess data (if not already done)
python src/train/preprocess_data.py

# 2. Train recognition model (development mode - quick test)
python src/train/train_recognition.py \
    --dataset food-101 \
    --dev-mode \
    --dev-samples 500 \
    --epochs 10 \
    --device mps

# 3. Full training (after dev mode validates)
python src/train/train_recognition.py \
    --dataset food-101 \
    --epochs 50 \
    --batch-size 32 \
    --device mps

# 4. Run inference
python src/train/inference_recognition.py \
    --image tests/test_food1.jpg \
    --model-path models/recognition/food-101_efficientnet_b0/best_f1.pt \
    --visualize
```

## Benefits

1. **Uses existing data effectively** - Food-101's 101k labeled images
2. **Leverages SAM2** - State-of-the-art segmentation without training
3. **Focuses effort on hard problem** - Food recognition, not segmentation
4. **Aligns with project goal** - Nutrition analysis
5. **Proven approach** - EfficientNet baselines: ~82% accuracy on Food-101

## Expected Results

**Development Mode** (500 samples, 10 epochs):
- Training time: ~5-10 minutes
- Val Accuracy: ~50-60%

**Full Training** (101k samples, 50 epochs):
- Training time: ~4-6 hours (M3 Mac)
- Val Accuracy: ~75-80%
- Top-5 Accuracy: ~90%+

## Next Steps

1. ✅ Architecture implemented
2. Run preprocessing (if needed)
3. **Run development training** - Validate approach
4. **Run full training** - Get production model
5. **Integrate with API** - Deploy to FastAPI backend

## Technical Details

**Model**: EfficientNet-B0 + 3-layer classification head
**Loss**: 0.7×CrossEntropy + 0.3×FocalLoss (with label smoothing)
**Metrics**: Accuracy, Precision, Recall, F1, Top-5 Accuracy
**Augmentation**: RandomResizedCrop, HorizontalFlip, ColorJitter, Cutout

## Files Structure

```
ml/
├── src/
│   ├── models/
│   │   ├── food_recognition.py          [NEW]
│   │   └── __init__.py                  [MODIFIED]
│   ├── data_process/
│   │   ├── food_labels.py               [NEW]
│   │   ├── classification_dataset.py    [NEW]
│   │   └── preprocessing.py             [MODIFIED]
│   ├── training/
│   │   ├── classification_trainer.py    [NEW]
│   │   ├── classification_metrics.py    [NEW]
│   │   └── classification_losses.py     [NEW]
│   ├── train/
│   │   ├── train_recognition.py         [NEW]
│   │   └── inference_recognition.py     [NEW]
│   └── config.py                        [MODIFIED]
├── RECOGNITION_ARCHITECTURE.md          [NEW]
└── ARCHITECTURE_CHANGE_SUMMARY.md       [NEW]
```

## Git Commands

```bash
# View changes
git status
git diff

# Commit
git add .
git commit -m "Implement food recognition architecture

- Add FoodRecognitionModel with EfficientNet-B0 backbone
- Create classification dataset and training pipeline
- Add SAM2 + Recognition inference pipeline
- Update preprocessing to extract food_class labels
- Add comprehensive documentation

This replaces segmentation training (which had no mask data)
with classification training (using Food-101's 101k labeled images)."

# Push to remote
git push -u origin feature/food-recognition-model
```

## Validation Plan

1. **Development Training** (10 min)
   - Run with --dev-mode
   - Verify training loop works
   - Check metrics make sense

2. **Full Training** (4-6 hours)
   - Train on full Food-101
   - Target: Val Accuracy > 75%
   - Save best checkpoint

3. **Inference Testing**
   - Test on sample images
   - Verify SAM2 + Recognition pipeline
   - Check nutrition lookup integration

4. **API Integration**
   - Add endpoint to FastAPI
   - Test with frontend
   - Deploy to production

## References

- **Full Documentation**: `RECOGNITION_ARCHITECTURE.md`
- **Training Script**: `src/train/train_recognition.py --help`
- **Inference Script**: `src/train/inference_recognition.py --help`
- **Model Code**: `src/models/food_recognition.py`

---

**Created**: 2025-10-31
**Branch**: `feature/food-recognition-model`
**Status**: Ready for testing
