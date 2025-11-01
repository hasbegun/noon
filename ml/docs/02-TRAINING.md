# Training Guide

Complete guide for training food recognition models to achieve 90-95% accuracy.

> 📖 **See also**:
> - [DATASETS_AND_INCREMENTAL_TRAINING.md](../DATASETS_AND_INCREMENTAL_TRAINING.md) - Dataset details
> - [HIGH_QUALITY_TRAINING_STRATEGY.md](../HIGH_QUALITY_TRAINING_STRATEGY.md) - Advanced strategies

---

## Quick Start

### Test Training (10 minutes)

```bash
# Quick test to verify everything works
python src/train/train_recognition.py \
    --dataset food-101 \
    --dev-mode \
    --dev-samples 100 \
    --epochs 2 \
    --device mps
```

### Production Training (16 hours)

```bash
# Full training for 90%+ accuracy
python src/train/train_recognition.py \
    --dataset food-101 \
    --backbone efficientnet_b3 \
    --image-size 300 \
    --batch-size 16 \
    --epochs 150 \
    --warmup-epochs 5 \
    --mixup --cutmix \
    --seed 42 \
    --device mps
```

---

## Available Datasets

| Dataset | Classes | Type | Best For |
|---------|---------|------|----------|
| **food-101** | 101 | Specific dishes | High accuracy |
| **nutrition5k** | 18 | Generic types | Nutrition data |
| **combined** | 115 | Both | Incremental training |

**Usage**:
```bash
--dataset food-101      # Train on Food-101
--dataset nutrition5k   # Train on Nutrition5k
--dataset combined      # Train on both (incremental)
```

---

## Training Parameters

### Model Architecture

```bash
--backbone efficientnet_b0  # 4.8M params, ~78% acc, fastest
--backbone efficientnet_b3  # 12M params, ~91% acc, recommended
--backbone efficientnet_b4  # 19M params, ~93% acc, slowest
--backbone resnet50         # 25M params, ~88% acc
```

### Training Settings

```bash
--epochs 150               # Number of epochs (150 for best results)
--batch-size 16            # Batch size (16 for EfficientNet-B3)
--lr 0.001                 # Learning rate
--warmup-epochs 5          # Warmup epochs for stability
--image-size 300           # Image size (300 for B3, 224 for B0)
```

### Data Augmentation

```bash
--mixup                    # Enable Mixup augmentation
--cutmix                   # Enable CutMix augmentation
--mixup-alpha 0.2          # Mixup strength
--cutmix-alpha 1.0         # CutMix strength
```

### Reproducibility

```bash
--seed 42                  # Random seed for reproducibility
```

### Development Mode

```bash
--dev-mode                 # Use small subset for quick testing
--dev-samples 500          # Number of samples in dev mode
```

---

## Incremental Training

Train on one dataset, then add another:

### Step 1: Train on Food-101

```bash
python src/train/train_recognition.py \
    --dataset combined \              # Use combined label space!
    --backbone efficientnet_b3 \
    --epochs 150 \
    --seed 42 \
    --device mps
```

### Step 2: Add Nutrition5k Data

```bash
python src/train/train_recognition.py \
    --dataset combined \
    --backbone efficientnet_b3 \
    --checkpoint models/recognition/combined_efficientnet_b3/last_checkpoint.pt \
    --epochs 50 \
    --lr 0.0001 \                     # Lower LR for fine-tuning
    --seed 42 \
    --device mps
```

**Key Points**:
- ✅ Use `--dataset combined` from the start
- ✅ Lower learning rate when adding new data
- ✅ Fewer epochs for later phases (50 vs 150)
- ✅ Use same seed for consistency

---

## Expected Results

### EfficientNet-B0 (Baseline)

```
Time: 6-8 hours
Expected accuracy: 75-80%
Top-5 accuracy: 95%+
```

### EfficientNet-B3 (Recommended)

```
Time: 12-16 hours
Expected accuracy: 90-93%
Top-5 accuracy: 98%+
```

### EfficientNet-B3 + Advanced Techniques

```
Time: 20-30 hours
Expected accuracy: 93-95%
Top-5 accuracy: 99%+

Techniques:
- Mixup/CutMix augmentation
- Two-stage training
- Higher resolution (384)
```

---

## Training Scripts

### High-Quality Training

```bash
# Use pre-configured script for best results
bash train_high_quality.sh
```

This runs:
- EfficientNet-B3 backbone
- 300×300 image size
- Mixup + CutMix augmentation
- 150 epochs with warmup
- Optimized batch size

### Two-Stage Training

```bash
# For even better results
bash train_two_stage.sh
```

**Stage 1**: Freeze backbone, train classifier (10 epochs)
**Stage 2**: Fine-tune entire model (100 epochs)

Expected: 94-96% accuracy

---

## Monitoring Training

### Training Output

```
Epoch 1/150 [Train]: 100%|████████| 4419/4419 [06:23<00:00]
  loss: 2.1234 cls_loss: 2.1234

Epoch 1/150 [Val]: 100%|████████| 946/946 [01:12<00:00]
  Metrics:
    accuracy: 0.7234  precision: 0.7156  recall: 0.7189  f1: 0.7172

✓ Best accuracy checkpoint saved: 0.7234
```

### Checkpoints Saved

```
models/recognition/food-101_efficientnet_b3/
├── best_accuracy.pt       # Best validation accuracy
├── best_f1.pt            # Best F1 score
├── last_checkpoint.pt    # Latest epoch (for resume)
└── label_mapping.json    # Class labels
```

---

## Auto-Resume

Training automatically resumes if it crashes:

```
📁 FOUND EXISTING CHECKPOINT
   Path: models/recognition/.../last_checkpoint.pt
   Last epoch: 45
   Best accuracy: 0.8945

   Options:
   1. Resume training (automatically in 10 seconds)
   2. Press Ctrl+C to cancel and start fresh
```

**To resume manually**:
```bash
python src/train/train_recognition.py \
    --checkpoint models/recognition/.../last_checkpoint.pt \
    --dataset food-101 \
    --epochs 150 \
    --device mps
```

---

## Troubleshooting

### Training Crashes

See [04-TROUBLESHOOTING.md](04-TROUBLESHOOTING.md#file-descriptor-leaks)

**Quick fix**:
```bash
# Increase file descriptor limit
ulimit -n 4096

# Restart training (auto-resumes)
python src/train/train_recognition.py ...
```

### Low Accuracy (<85%)

**Common causes**:
- Not enough epochs (need 150+)
- Model too small (use EfficientNet-B3)
- No augmentation (add --mixup --cutmix)

**Solution**:
```bash
python src/train/train_recognition.py \
    --dataset food-101 \
    --backbone efficientnet_b3 \  # Larger model
    --epochs 150 \                 # More epochs
    --mixup --cutmix \            # Augmentation
    --device mps
```

### Out of Memory

**Reduce batch size**:
```bash
--batch-size 8   # Instead of 16
```

**Or use gradient accumulation** (coming soon)

---

## Performance Optimization

### macOS/Apple Silicon

- ✅ Uses MPS (Metal Performance Shaders) automatically
- ✅ File descriptor limits handled
- ✅ Memory optimized for M-series chips

### NVIDIA GPUs

```bash
--device cuda
--batch-size 32  # Can use larger batch size
```

### CPU Only

```bash
--device cpu
--batch-size 4   # Smaller batch size
--num-workers 0  # No multiprocessing
```

---

## Advanced Topics

### Ensemble Training

Train multiple models with different seeds:

```bash
# Model 1
python src/train/train_recognition.py --seed 42 ...

# Model 2
python src/train/train_recognition.py --seed 123 ...

# Model 3
python src/train/train_recognition.py --seed 456 ...
```

Ensemble improves accuracy by 2-4%.

### Custom Learning Rate Schedule

```bash
--warmup-epochs 5    # Linear warmup
# Cosine annealing automatically applied
```

### Freeze Backbone (Fine-tuning)

```bash
--freeze-backbone    # Only train classifier head
--lr 0.001          # Higher LR for classifier
```

---

## Next Steps

After training:

1. **Test your model**: See [03-TESTING.md](03-TESTING.md)
2. **Review results**: Check `models/recognition/.../`
3. **Deploy**: See [06-DEPLOYMENT.md](06-DEPLOYMENT.md)

---

## Additional Resources

- **[HIGH_QUALITY_TRAINING_STRATEGY.md](../HIGH_QUALITY_TRAINING_STRATEGY.md)** - Complete training roadmap to 95%+
- **[DATASETS_AND_INCREMENTAL_TRAINING.md](../DATASETS_AND_INCREMENTAL_TRAINING.md)** - Dataset details and incremental training

---

**Ready to train!** 🚀 Start with the quick test, then move to production training.
