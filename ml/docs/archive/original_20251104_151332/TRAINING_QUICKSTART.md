# Training Quick Start Guide

## Development Mode (Quick Iteration)

For fast iteration and monitoring F1 scores, use **development mode** with a small subset of data:

```bash
cd ml

# Quick training with 100 samples (fast epochs!)
python src/train/train.py --dev-mode --epochs 10

# Adjust number of dev samples
python src/train/train.py --dev-mode --dev-samples 200 --epochs 10

# With custom batch size
python src/train/train.py --dev-mode --batch-size 4 --epochs 10
```

### What Dev Mode Does:
- ✅ Uses only **100 training samples** (instead of full dataset)
- ✅ Uses only **50 validation samples**
- ✅ **Much faster epochs** (~seconds instead of minutes)
- ✅ Shows **F1 score** for each epoch
- ✅ Perfect for debugging and monitoring metrics

### Expected Output:

```
🔧 DEVELOPMENT MODE ENABLED
   Using only 100 training samples
   Using only 50 validation samples
   For full training, remove --dev-mode flag

Epoch 1/10: 100% ████████ loss=0.42
Epoch 1/10 Complete - Train Loss: 0.42, Val Loss: 0.38 | Train F1: 0.15, Val F1: 0.18
Train Metrics - Precision: 0.1234 | Recall: 0.1890 | F1: 0.1500 | Iou: 0.0812 | Accuracy: 0.8932
Val Metrics - Precision: 0.1456 | Recall: 0.2234 | F1: 0.1789 | Iou: 0.0981 | Accuracy: 0.9012

Epoch 2/10: 100% ████████ loss=0.35
Epoch 2/10 Complete - Train Loss: 0.35, Val Loss: 0.32 | Train F1: 0.23, Val F1: 0.25
...
```

**Watch for**:
- ✅ **Loss decreasing** (not going to 0.0!)
- ✅ **F1 score increasing** over epochs
- ✅ **No collapse warnings**

## Full Training (Production)

Once dev mode looks good, train on full dataset:

```bash
cd ml

# Full training with all data
python src/train/train.py --epochs 50

# With custom settings
python src/train/train.py \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-4 \
  --num-workers 4
```

## Monitoring Metrics

The trainer now shows detailed metrics for each epoch:

### Metrics Displayed:
- **Loss**: Combined loss (BCE + Dice + Focal)
- **F1 Score**: Main metric for segmentation quality
- **Precision**: How many predicted pixels are correct
- **Recall**: How many ground truth pixels are found
- **IoU**: Intersection over Union
- **Accuracy**: Overall pixel accuracy

### Best Model Selection:
- Model is saved when **F1 score improves** (not just loss!)
- File: `models/segmentation/best_model.pt`

## Quick Commands

### Development (Fast)
```bash
# Quick 10 epochs with metrics
python src/train/train.py --dev-mode --epochs 10

# Very quick test (50 samples, 5 epochs)
python src/train/train.py --dev-mode --dev-samples 50 --epochs 5

# Monitor specific metrics
python src/train/train.py --dev-mode --epochs 10 2>&1 | grep "F1:"
```

### Production (Full Data)
```bash
# Standard full training
python src/train/train.py --epochs 50

# Resume from checkpoint
python src/train/train.py --epochs 100 --checkpoint models/segmentation/last_checkpoint.pt

# Custom configuration
python src/train/train.py \
  --epochs 50 \
  --batch-size 16 \
  --lr 5e-5 \
  --num-workers 8
```

## Training Workflow

### 1. Start with Dev Mode
```bash
# Quick test to verify training works
python src/train/train.py --dev-mode --epochs 5
```

**Check**:
- Loss decreases ✓
- F1 score increases ✓
- No warnings about prediction collapse ✓

### 2. Tune Hyperparameters (Still Dev Mode)
```bash
# Try different learning rates
python src/train/train.py --dev-mode --epochs 10 --lr 1e-5
python src/train/train.py --dev-mode --epochs 10 --lr 5e-5

# Try different batch sizes
python src/train/train.py --dev-mode --epochs 10 --batch-size 4
python src/train/train.py --dev-mode --epochs 10 --batch-size 16
```

### 3. Full Training
```bash
# Once happy with dev mode results
python src/train/train.py --epochs 50 --batch-size 8 --lr 1e-4
```

## Understanding Metrics

### F1 Score
- **0.0 - 0.3**: Poor segmentation
- **0.3 - 0.5**: Fair segmentation
- **0.5 - 0.7**: Good segmentation ✓
- **0.7 - 0.9**: Very good segmentation ✓✓
- **0.9 - 1.0**: Excellent segmentation ✓✓✓

### What to Expect:
- **Epoch 1-5**: F1 ~ 0.1-0.3 (model learning)
- **Epoch 10-20**: F1 ~ 0.3-0.5 (getting better)
- **Epoch 30-50**: F1 ~ 0.5-0.7+ (good performance)

### Warning Signs:
- ❌ **Loss = 0.0**: Model collapsed (should not happen with new loss!)
- ❌ **F1 stays at 0.0**: No valid masks in dataset
- ❌ **F1 decreasing**: Overfitting or learning rate too high

## Troubleshooting

### Issue: Epochs are too slow in dev mode
```bash
# Use fewer samples
python src/train/train.py --dev-mode --dev-samples 50 --epochs 5

# Reduce batch size
python src/train/train.py --dev-mode --batch-size 2
```

### Issue: F1 score stays at 0.0
```bash
# Check if dataset has masks
python debug_masks.py

# If no masks, you need segmentation annotations
# Or generate pseudo-masks first
```

### Issue: Loss goes to 0.0 (shouldn't happen!)
```bash
# Verify you have the new loss function
grep "CombinedSegmentationLoss" src/training/trainer.py

# If not, you may need to pull latest changes
```

### Issue: Out of memory
```bash
# Reduce batch size
python src/train/train.py --dev-mode --batch-size 2

# Reduce image size (in config)
# Or use fewer samples
python src/train/train.py --dev-mode --dev-samples 50
```

## Files and Outputs

### Checkpoints (in `models/segmentation/`)
- `best_model.pt` - Best model by F1 score
- `last_checkpoint.pt` - Latest checkpoint (for resume)
- `checkpoint_epoch_N.pt` - Periodic checkpoints (every 10 epochs)
- `final_model.pt` - Final model after all epochs

### Logs
Training logs show:
```
Epoch 5/50 Complete - Train Loss: 0.32, Val Loss: 0.28 | Train F1: 0.45, Val F1: 0.48
Train Metrics - Precision: 0.4234 | Recall: 0.4789 | F1: 0.4500 | Iou: 0.2893 | Accuracy: 0.9234
Val Metrics - Precision: 0.4556 | Recall: 0.5123 | F1: 0.4823 | Iou: 0.3178 | Accuracy: 0.9312
✓ Saved best model (F1: 0.4823, IoU: 0.3178)
```

## Dev vs Full Training Comparison

| Aspect | Dev Mode | Full Training |
|--------|----------|---------------|
| **Samples** | 100 train, 50 val | Full dataset |
| **Epoch Time** | 5-30 seconds | 5-30 minutes |
| **Total Time (10 epochs)** | 1-5 minutes | 1-5 hours |
| **Use Case** | Debug, tune, verify | Final training |
| **F1 Accuracy** | Approximate | Actual |
| **Command** | `--dev-mode` | No flag |

## Summary

### Quick Start:
```bash
# 1. Test training works (2 minutes)
python src/train/train.py --dev-mode --epochs 5

# 2. Monitor F1 scores (5 minutes)
python src/train/train.py --dev-mode --epochs 10

# 3. Full training (2+ hours)
python src/train/train.py --epochs 50
```

### Key Improvements:
- ✅ **F1 Score** shown every epoch
- ✅ **Dev mode** for quick iteration
- ✅ **Combined loss** prevents collapse
- ✅ **Detailed metrics** (Precision, Recall, IoU)
- ✅ **Best model** saved by F1 score

You can now quickly iterate and see F1 scores! Start with `--dev-mode` to verify everything works, then do full training.
