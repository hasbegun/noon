# Checkpoint and Resume Training Guide

## Overview

Training now includes automatic checkpoint saving and resume functionality. If training is interrupted, you can resume from where it left off without losing progress.

## How Checkpoints Work

### Automatic Checkpoint Saving

After **every epoch**, three types of checkpoints are saved:

1. **`last_checkpoint.pt`** - Always updated after each epoch (for resume)
2. **`best_model.pt`** - Saved when validation loss improves
3. **`checkpoint_epoch_N.pt`** - Saved every 10 epochs (10, 20, 30, etc.)

### What Gets Saved

Each checkpoint contains:
- ✅ Model weights
- ✅ Optimizer state (learning rates, momentum, etc.)
- ✅ Scheduler state (learning rate schedule)
- ✅ Current epoch number
- ✅ Best validation loss so far
- ✅ Training/validation loss history
- ✅ Mixed precision scaler state (if using AMP)

## Resuming Training

### Method 1: Automatic Resume (Interactive)

Simply run the same training command:

```bash
cd ml
conda activate noon2

python src/train/train.py --epochs 50 --batch-size 8 --device mps
```

If a checkpoint exists, you'll see:
```
WARNING: Found existing checkpoint: models/segmentation/last_checkpoint.pt
WARNING: To resume training, use: --checkpoint models/segmentation/last_checkpoint.pt
WARNING: To start fresh, delete the checkpoint or use a new model name
Resume from last checkpoint? [y/N]:
```

Type `y` to resume, or `N` to start fresh.

### Method 2: Manual Resume with --checkpoint

Explicitly specify which checkpoint to resume from:

```bash
# Resume from last checkpoint
python src/train/train.py \
    --epochs 50 \
    --batch-size 8 \
    --device mps \
    --checkpoint models/segmentation/last_checkpoint.pt

# Or resume from a specific epoch checkpoint
python src/train/train.py \
    --epochs 50 \
    --batch-size 8 \
    --device mps \
    --checkpoint models/segmentation/checkpoint_epoch_10.pt

# Or resume from best model
python src/train/train.py \
    --epochs 50 \
    --batch-size 8 \
    --device mps \
    --checkpoint models/segmentation/best_model.pt
```

### Method 3: Using Make

```bash
cd ml

# Resume from last checkpoint
make train-resume EPOCHS=50 BATCH_SIZE=8
```

## What Happens During Resume

When resuming from a checkpoint:

```
INFO | Loading checkpoint: models/segmentation/last_checkpoint.pt
INFO | ✓ Checkpoint loaded successfully:
INFO |   - Resuming from epoch 2
INFO |   - Best val loss so far: 0.0123
INFO |   - Training history: 1 epochs
INFO | Resuming training from epoch 2/50
INFO | Starting Epoch 2/50 - Training...
Epoch 2/50:   0%|  | 0/8837 [00:00<?, ?it/s]
```

**Training continues from where it stopped!**

## Example Scenarios

### Scenario 1: Training Interrupted After 1 Epoch

**What happened:**
- Training completed epoch 1
- Saved `last_checkpoint.pt` with epoch=1
- Training stopped (Ctrl+C or system crash)

**To resume:**
```bash
python src/train/train.py --epochs 50 --batch-size 8 --device mps
# Select 'y' when prompted, or use --checkpoint flag
```

**Result:**
- Loads checkpoint from epoch 1
- Resumes training at epoch 2
- Continues through epochs 2, 3, 4, ..., 50

### Scenario 2: Want to Continue Training Beyond 50 Epochs

**What happened:**
- Trained for 50 epochs
- Want to train for 100 total epochs

**To continue:**
```bash
python src/train/train.py \
    --epochs 100 \
    --batch-size 8 \
    --device mps \
    --checkpoint models/segmentation/last_checkpoint.pt
```

**Result:**
- Loads checkpoint from epoch 50
- Continues training epochs 51-100

### Scenario 3: Start Fresh (Ignore Checkpoint)

**Option A: Delete checkpoint**
```bash
rm models/segmentation/last_checkpoint.pt
python src/train/train.py --epochs 50 --batch-size 8 --device mps
```

**Option B: Say 'N' when prompted**
```bash
python src/train/train.py --epochs 50 --batch-size 8 --device mps
# Type 'N' when asked to resume
```

**Result:**
- Training starts from epoch 0
- Previous checkpoint is overwritten

## Checkpoint File Locations

Default location: `ml/models/segmentation/`

```
ml/models/segmentation/
├── last_checkpoint.pt        # Most recent epoch (auto-updated)
├── best_model.pt             # Best validation loss
├── checkpoint_epoch_10.pt    # Snapshot at epoch 10
├── checkpoint_epoch_20.pt    # Snapshot at epoch 20
├── checkpoint_epoch_30.pt    # Snapshot at epoch 30
└── final_model.pt            # Saved when training completes
```

## Verifying Checkpoints

### List Available Checkpoints

```bash
ls -lh ml/models/segmentation/*.pt
```

### Check Checkpoint Contents

```python
import torch

checkpoint = torch.load("models/segmentation/last_checkpoint.pt")

print(f"Epoch: {checkpoint['epoch']}")
print(f"Best val loss: {checkpoint['best_val_loss']:.4f}")
print(f"Training history: {len(checkpoint['train_losses'])} epochs")
print(f"Config: {checkpoint['config']}")
```

## Common Issues & Solutions

### Issue: "Checkpoint not found"
**Solution**: Check the path. Checkpoints are in `models/segmentation/`, not project root.

```bash
ls models/segmentation/last_checkpoint.pt
```

### Issue: "Training starts from epoch 0 even with checkpoint"
**Solution**: This is now fixed! Training will properly resume from the checkpoint epoch.

### Issue: "Want to change hyperparameters when resuming"
**Warning**: Changing some parameters mid-training can cause issues:

Safe to change:
- ✅ `--epochs` (extend training)
- ✅ `--num-workers` (data loading)

Requires caution:
- ⚠️ `--batch-size` (may affect training stability)
- ⚠️ `--lr` (learning rate already scheduled by loaded optimizer)

Not recommended:
- ❌ `--model-type` (model architecture mismatch)
- ❌ `--device` (weights may not transfer correctly)

### Issue: "Lost progress because training stopped"
**Solution**: Now every epoch auto-saves! You'll never lose more than 1 epoch of progress.

### Issue: "Running out of disk space with checkpoints"
**Solution**:
```bash
# Keep only last and best checkpoints
cd ml/models/segmentation
rm checkpoint_epoch_*.pt  # Delete periodic checkpoints

# Or keep only last checkpoint
rm best_model.pt checkpoint_epoch_*.pt
```

## Best Practices

### 1. Always Use caffeinate for Long Training
```bash
caffeinate python src/train/train.py --epochs 50 --batch-size 8 --device mps
```
Prevents Mac from sleeping during training.

### 2. Monitor Training
Open another terminal to monitor progress:
```bash
# Watch for new checkpoints
watch -n 60 'ls -lh ml/models/segmentation/*.pt'

# Watch logs
tail -f logs/training.log
```

### 3. Backup Important Checkpoints
```bash
# After finding a good checkpoint, back it up
cp models/segmentation/best_model.pt models/segmentation/best_model_backup_$(date +%Y%m%d).pt
```

### 4. Test Resume Before Long Training
```bash
# Train for 2 epochs
python src/train/train.py --epochs 2 --batch-size 8 --device mps

# Stop and resume
python src/train/train.py --epochs 5 --batch-size 8 --device mps --checkpoint models/segmentation/last_checkpoint.pt

# Verify it continues from epoch 3
```

## Make Commands

Add to Makefile:
```makefile
# Resume training from last checkpoint
train-resume:
	@echo "Resuming training from last checkpoint..."
	$(CONDA_RUN) python src/train/train.py \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--device $(DEVICE) \
		--checkpoint models/segmentation/last_checkpoint.pt

# Clean checkpoints (keep best)
clean-checkpoints:
	@echo "Cleaning old checkpoints (keeping best_model.pt)..."
	rm -f models/segmentation/checkpoint_epoch_*.pt
	rm -f models/segmentation/last_checkpoint.pt
```

Usage:
```bash
make train-resume EPOCHS=50 BATCH_SIZE=8
make clean-checkpoints
```

## Summary

| Feature | Status | Description |
|---------|--------|-------------|
| Auto-save after each epoch | ✅ | Never lose progress |
| Resume from checkpoint | ✅ | Continue where you left off |
| Keep best model | ✅ | Best validation loss saved |
| Periodic snapshots | ✅ | Every 10 epochs |
| State preservation | ✅ | Model, optimizer, scheduler all saved |
| Interactive resume | ✅ | Prompted when checkpoint exists |
| Manual resume | ✅ | Via --checkpoint flag |

**Your training is now safe from interruptions!** 🛡️

---

**Last Updated**: 2025-10-29
**Related Files**:
- `src/training/trainer.py` (checkpoint save/load)
- `src/train/train.py` (resume logic)
