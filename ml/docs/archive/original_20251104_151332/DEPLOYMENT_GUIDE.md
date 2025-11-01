# Deployment Guide for Remote Training (192.168.14.12)

## Pre-Training Review & Remote Setup

---

## 1. Configuration Review

### Current Setup Summary

**Model Configuration:**
- Backbone: EfficientNet-B3 (12M parameters)
- Image size: 300×300
- Num classes: 101 (Food-101)
- Dropout: 0.3

**Training Configuration:**
- Epochs: 150
- Batch size: 16
- Learning rate: 0.001
- Warmup epochs: 5
- Scheduler: Cosine annealing with linear warmup

**Data Augmentation:**
- Mixup: Enabled (alpha=0.2)
- CutMix: Enabled (alpha=1.0)
- Standard: RandomResizedCrop, HorizontalFlip, ColorJitter, etc.

**Memory Management:**
- MPS cache clearing: Every 5 batches (training) / 3 batches (validation)
- Immediate CPU transfer for metrics
- Explicit tensor deletion
- Environment variables set for overflow handling

**Expected Results:**
- Training time: 25-30 hours
- Final accuracy: 93-95%
- Memory usage: ~24 GB

---

## 2. Remote Machine Requirements

### Hardware Requirements

**Minimum:**
- GPU: Apple Silicon (M1/M2/M3) with 24+ GB unified memory
  - OR NVIDIA GPU with 16+ GB VRAM
- RAM: 32+ GB
- Storage: 50+ GB free space

**Optimal:**
- GPU: M2 Max/Ultra or M3 Max/Ultra
  - OR NVIDIA A100/V100 with 32+ GB
- RAM: 64+ GB
- Storage: 100+ GB SSD

### Software Requirements

```bash
# Python
python >= 3.10

# PyTorch
torch >= 2.0.0 (with MPS or CUDA support)

# Required packages
pip install -r requirements.txt
```

**Verify GPU availability:**
```bash
# For Apple Silicon (MPS)
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# For NVIDIA (CUDA)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 3. File Transfer to Remote Machine

### Method 1: rsync (Recommended)

```bash
# From your current machine, sync to remote
rsync -avz --progress \
    --exclude='models/' \
    --exclude='data/raw/' \
    --exclude='*.pyc' \
    --exclude='__pycache__/' \
    --exclude='.git/' \
    ~/projects/noon2/ml/ \
    user@192.168.14.12:~/noon2/ml/
```

### Method 2: Git (Alternative)

```bash
# On current machine - commit changes
cd ~/projects/noon2/ml
git add .
git commit -m "Add high-quality training implementation (Option B)"
git push

# On remote machine - pull changes
ssh user@192.168.14.12
cd ~/noon2/ml
git pull
```

### Method 3: SCP (Simple transfer)

```bash
# Transfer entire directory
scp -r ~/projects/noon2/ml/ user@192.168.14.12:~/noon2/
```

---

## 4. Data Setup on Remote Machine

### Check Data Availability

```bash
# SSH into remote machine
ssh user@192.168.14.12

# Check if data exists
cd ~/noon2/ml
ls -lh data/processed/food-101/

# Expected structure:
# data/processed/food-101/
#   ├── train.parquet
#   ├── val.parquet
#   └── images/
#       ├── train/
#       └── val/
```

### If Data Missing - Transfer

**Option A: rsync from current machine**
```bash
# From current machine
rsync -avz --progress \
    ~/projects/noon2/ml/data/processed/ \
    user@192.168.14.12:~/noon2/ml/data/processed/
```

**Option B: Download directly on remote**
```bash
# On remote machine
cd ~/noon2/ml
python src/data_process/preprocessing.py --dataset food-101
```

### Verify Data

```bash
# On remote machine
python -c "
import pandas as pd
from pathlib import Path

train = pd.read_parquet('data/processed/food-101/train.parquet')
val = pd.read_parquet('data/processed/food-101/val.parquet')

print(f'Train samples: {len(train)}')
print(f'Val samples: {len(val)}')
print(f'Train classes: {train.label.nunique()}')
"

# Expected output:
# Train samples: 70700
# Val samples: 15150
# Train classes: 101
```

---

## 5. Environment Setup on Remote

### Install Dependencies

```bash
# On remote machine
cd ~/noon2/ml

# Create virtual environment (if not exists)
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# OR: venv\Scripts\activate  # On Windows

# Install requirements
pip install -r requirements.txt

# Verify installations
python -c "
import torch
import torchvision
import albumentations
from loguru import logger

print(f'PyTorch: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print('All imports successful!')
"
```

### Set Environment Variables

```bash
# Add to ~/.bashrc or ~/.zshrc on remote machine
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_MPS_ALLOCATOR_POLICY=garbage_collection

# Apply immediately
source ~/.bashrc  # or ~/.zshrc
```

---

## 6. Pre-Flight Checklist

### Run All Checks

```bash
# On remote machine
cd ~/noon2/ml

# 1. Check data
echo "=== Checking data ==="
python -c "
import pandas as pd
train = pd.read_parquet('data/processed/food-101/train.parquet')
val = pd.read_parquet('data/processed/food-101/val.parquet')
print(f'✓ Train: {len(train)} samples')
print(f'✓ Val: {len(val)} samples')
"

# 2. Check GPU
echo "=== Checking GPU ==="
python -c "
import torch
print(f'✓ MPS available: {torch.backends.mps.is_available()}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
"

# 3. Check packages
echo "=== Checking packages ==="
python -c "
import torch
import albumentations
from loguru import logger
print('✓ All packages imported')
"

# 4. Quick feature test
echo "=== Testing features ==="
./test_high_quality_features.sh

# Expected: All tests pass ✓
```

---

## 7. Training Configuration Files

### Review Training Scripts

**Single-Stage Training:** `train_high_quality.sh`
```bash
#!/bin/bash
# Configuration
DATASET="food-101"
BACKBONE="efficientnet_b3"
IMAGE_SIZE=300
BATCH_SIZE=16
EPOCHS=150
WARMUP_EPOCHS=5
LEARNING_RATE=0.001
NUM_WORKERS=4

# MPS memory management
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_MPS_ALLOCATOR_POLICY=garbage_collection

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
```

**Two-Stage Training:** `train_two_stage.sh`
- Stage 1: 10 epochs, frozen backbone, LR=0.001
- Stage 2: 100 epochs, full fine-tune, LR=0.0001

---

## 8. Recommended Configuration Adjustments

### If Remote Has More Memory (64+ GB)

```bash
# Can increase batch size for faster training
BATCH_SIZE=24  # Instead of 16

# Or increase image size for better accuracy
IMAGE_SIZE=320  # Instead of 300
```

### If Remote Has NVIDIA GPU (CUDA)

```bash
# In training scripts, change:
--device mps
# To:
--device cuda

# Can also enable mixed precision for speedup
--mixed-precision  # If implemented
```

### If Remote Has Less Memory (<32 GB)

```bash
# Reduce batch size
BATCH_SIZE=12  # Instead of 16

# Or reduce image size
IMAGE_SIZE=256  # Instead of 300
```

---

## 9. Running Training on Remote

### Option A: Single-Stage (Recommended)

```bash
# SSH into remote
ssh user@192.168.14.12

# Navigate to project
cd ~/noon2/ml

# Activate environment
source venv/bin/activate

# Run training (in tmux/screen for persistence)
tmux new -s training
./train_high_quality.sh

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

### Option B: Two-Stage (Best Quality)

```bash
# Same setup as above, but run:
./train_two_stage.sh
```

### Option C: Background with Logging

```bash
# Run in background with full logging
nohup ./train_high_quality.sh > training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Monitor progress
tail -f training_*.log

# Check process
ps aux | grep train_recognition
```

---

## 10. Monitoring Training Remotely

### From Your Current Machine

**Monitor via SSH:**
```bash
# Watch logs in real-time
ssh user@192.168.14.12 "tail -f ~/noon2/ml/training_*.log"

# Check GPU usage
ssh user@192.168.14.12 "watch -n 5 'ps aux | grep python'"

# Check saved checkpoints
ssh user@192.168.14.12 "ls -lht ~/noon2/ml/models/recognition/food-101_efficientnet_b3/"
```

**Set up periodic status checks:**
```bash
# Create a monitor script on remote
cat > ~/monitor_training.sh << 'EOF'
#!/bin/bash
echo "=== Training Status ==="
echo "Time: $(date)"
echo ""
echo "GPU Process:"
ps aux | grep train_recognition | grep -v grep
echo ""
echo "Latest checkpoints:"
ls -lht ~/noon2/ml/models/recognition/food-101_efficientnet_b3/ | head -5
echo ""
echo "Latest log:"
tail -n 20 ~/noon2/ml/training_*.log | grep "Epoch"
EOF

chmod +x ~/monitor_training.sh

# Run from current machine
ssh user@192.168.14.12 "~/monitor_training.sh"
```

---

## 11. Expected Training Timeline

### Single-Stage (150 epochs)

**Milestones:**
- **Hour 0-2**: Warmup phase (epochs 1-5)
  - Loss: 6.0 → 4.5
  - Accuracy: 5% → 15%

- **Hour 2-10**: Rapid learning (epochs 6-50)
  - Loss: 4.5 → 2.5
  - Accuracy: 15% → 60%

- **Hour 10-20**: Refinement (epochs 51-120)
  - Loss: 2.5 → 1.5
  - Accuracy: 60% → 85%

- **Hour 20-30**: Fine-tuning (epochs 121-150)
  - Loss: 1.5 → 1.0
  - Accuracy: 85% → **93-95%**

**Checkpoints:**
- Best accuracy model saved when validation accuracy improves
- Best F1 model saved when F1 score improves
- Last checkpoint saved every epoch

---

## 12. Post-Training Validation

### After Training Completes

```bash
# On remote machine
cd ~/noon2/ml

# Check final models
ls -lh models/recognition/food-101_efficientnet_b3/

# View training history
python -c "
import torch
checkpoint = torch.load('models/recognition/food-101_efficientnet_b3/last_checkpoint.pt')
history = checkpoint['history']

print('Training History Summary:')
print(f\"Best Val Accuracy: {checkpoint['best_val_acc']:.4f}\")
print(f\"Best Val F1: {checkpoint['best_val_f1']:.4f}\")
print(f\"Final Train Loss: {history['train_loss'][-1]:.4f}\")
print(f\"Final Val Loss: {history['val_loss'][-1]:.4f}\")
print(f\"Final Train Accuracy: {history['train_acc'][-1]:.4f}\")
print(f\"Final Val Accuracy: {history['val_acc'][-1]:.4f}\")
"
```

### Download Trained Models

```bash
# From current machine
scp -r user@192.168.14.12:~/noon2/ml/models/recognition/food-101_efficientnet_b3/ \
    ~/projects/noon2/ml/models/recognition/
```

---

## 13. Troubleshooting Guide

### Issue: Out of Memory Error

**Solution 1: Reduce batch size**
```bash
# Edit train_high_quality.sh
BATCH_SIZE=12  # or 8
```

**Solution 2: Reduce image size**
```bash
IMAGE_SIZE=256  # or 224
```

**Solution 3: Disable one augmentation**
```bash
# Remove either --mixup or --cutmix
# Keep only one
```

### Issue: Training Too Slow

**Check:**
```bash
# Verify GPU is being used
python -c "
import torch
print(f'Device: {torch.device(\"mps\" if torch.backends.mps.is_available() else \"cpu\")}')
"

# Check if other processes using GPU
ps aux | grep python
```

**Solution:**
- Ensure no other training jobs running
- Check NUM_WORKERS (reduce if CPU bottleneck)
- Monitor with Activity Monitor

### Issue: Connection Lost During Training

**Prevention:**
```bash
# Always use tmux or screen
tmux new -s training

# Or use nohup
nohup ./train_high_quality.sh > training.log 2>&1 &
```

**Recovery:**
```bash
# Check if training still running
ps aux | grep train_recognition

# If running, reattach to tmux
tmux attach -t training

# If stopped, resume from checkpoint
python src/train/train_recognition.py \
    --checkpoint models/recognition/food-101_efficientnet_b3/last_checkpoint.pt \
    ... (same args as before)
```

---

## 14. Quick Reference Commands

### Setup
```bash
# Transfer files
rsync -avz ~/projects/noon2/ml/ user@192.168.14.12:~/noon2/ml/

# SSH and setup
ssh user@192.168.14.12
cd ~/noon2/ml
source venv/bin/activate
```

### Start Training
```bash
# In tmux
tmux new -s training
./train_high_quality.sh

# Detach: Ctrl+B, D
```

### Monitor
```bash
# From local machine
ssh user@192.168.14.12 "tail -f ~/noon2/ml/training_*.log | grep Epoch"
```

### Check Status
```bash
ssh user@192.168.14.12 "ps aux | grep train_recognition"
```

### Download Models
```bash
scp -r user@192.168.14.12:~/noon2/ml/models/recognition/ \
    ~/projects/noon2/ml/models/
```

---

## 15. Final Pre-Training Checklist

**On Remote Machine (192.168.14.12):**

- [ ] Code transferred and up-to-date
- [ ] Data available and verified (70,700 train / 15,150 val)
- [ ] Dependencies installed (torch, albumentations, etc.)
- [ ] GPU available (MPS or CUDA)
- [ ] Environment variables set
- [ ] Test script passed (`./test_high_quality_features.sh`)
- [ ] Training script reviewed and adjusted
- [ ] tmux/screen session ready
- [ ] Monitoring setup configured
- [ ] Sufficient disk space (50+ GB free)

**Ready to train!**

---

## 16. Command to Start

```bash
# On remote machine (192.168.14.12)
cd ~/noon2/ml
source venv/bin/activate
tmux new -s training
./train_high_quality.sh

# Detach: Ctrl+B, then D
# Expected time: 25-30 hours
# Expected accuracy: 93-95%
```

---

**Created**: 2025-10-31
**Target Machine**: 192.168.14.12
**Estimated Training Time**: 25-30 hours
**Expected Accuracy**: 93-95%
**Status**: Ready for deployment
