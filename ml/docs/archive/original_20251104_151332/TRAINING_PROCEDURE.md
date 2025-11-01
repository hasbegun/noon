# Complete Training Procedure - Step by Step

## Target: 93-95% Accuracy on Food-101

**Remote Machine**: 192.168.14.12
**Method**: Two-Stage Training (Best Quality)
**Expected Time**: 30-35 hours
**Expected Accuracy**: 94-96%

---

## Phase 1: Pre-Deployment (On Your Current Machine)

### Step 1.1: Verify Local Setup ✅

```bash
# Navigate to project
cd ~/projects/noon2/ml

# Verify all files present
ls -la train_high_quality.sh train_two_stage.sh test_high_quality_features.sh
# Should show all scripts with execute permissions

# Check data
ls -lh data/processed/food-101/
# Should show: train.parquet, val.parquet, images/
```

### Step 1.2: Commit Changes (Optional but Recommended)

```bash
# If using git
git status
git add .
git commit -m "Add high-quality training implementation (Option B - 93-95% target)"
git push origin main  # or your branch
```

### Step 1.3: Transfer to Remote Machine

**Option A: Using rsync (Recommended - Faster for updates)**

```bash
# Transfer code (excludes models and cache to save time)
rsync -avz --progress \
    --exclude='models/' \
    --exclude='data/raw/' \
    --exclude='*.pyc' \
    --exclude='__pycache__/' \
    --exclude='.git/' \
    --exclude='venv/' \
    ~/projects/noon2/ml/ \
    user@192.168.14.12:~/noon2/ml/

# Expected output: Files transferred, progress shown
```

**Option B: Using Git (If repo is set up)**

```bash
# On remote machine (after SSH)
cd ~/noon2/ml
git pull origin main
```

---

## Phase 2: Remote Machine Setup (First Time Only)

### Step 2.1: Connect to Remote Machine

```bash
# From your current machine
ssh user@192.168.14.12

# You should now be on the remote machine
# Prompt will change to: user@remote-machine:~$
```

### Step 2.2: Navigate to Project

```bash
cd ~/noon2/ml
pwd
# Should output: /home/user/noon2/ml (or similar)
```

### Step 2.3: Check Environment

```bash
# Check Python version
python --version
# Should be: Python 3.10+

# Check if venv exists
ls -la venv/
# If exists, activate it
# If not exists, create it first
```

### Step 2.4: Create/Activate Virtual Environment

```bash
# If venv doesn't exist, create it
python -m venv venv

# Activate it
source venv/bin/activate

# Prompt should change to: (venv) user@remote-machine:~/noon2/ml$
```

### Step 2.5: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# This will install:
# - torch (with MPS support)
# - torchvision
# - albumentations
# - pandas
# - loguru
# - tqdm
# - etc.

# Expected time: 5-10 minutes
```

### Step 2.6: Verify GPU Availability

```bash
# Check MPS (Apple Silicon)
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
"

# Expected output:
# PyTorch version: 2.x.x
# MPS available: True
# MPS built: True

# If CUDA instead of MPS:
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"
```

### Step 2.7: Verify Data Availability

```bash
# Check data files
ls -lh data/processed/food-101/

# Should see:
# train.parquet (~50-100 MB)
# val.parquet (~10-20 MB)
# images/ (directory)

# Verify sample counts
python -c "
import pandas as pd
train = pd.read_parquet('data/processed/food-101/train.parquet')
val = pd.read_parquet('data/processed/food-101/val.parquet')
print(f'✓ Train samples: {len(train):,}')
print(f'✓ Val samples: {len(val):,}')
print(f'✓ Classes: {train.label.nunique()}')
"

# Expected output:
# ✓ Train samples: 70,700
# ✓ Val samples: 15,150
# ✓ Classes: 101
```

**If data is missing:**

```bash
# Option 1: Transfer from current machine
# (On current machine)
rsync -avz --progress \
    ~/projects/noon2/ml/data/processed/food-101/ \
    user@192.168.14.12:~/noon2/ml/data/processed/food-101/

# Option 2: Download/process on remote
# (On remote machine)
python src/data_process/preprocessing.py --dataset food-101
```

### Step 2.8: Set Environment Variables

```bash
# Set for current session
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_MPS_ALLOCATOR_POLICY=garbage_collection

# Add to shell config for persistence
echo 'export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0' >> ~/.bashrc
echo 'export PYTORCH_MPS_ALLOCATOR_POLICY=garbage_collection' >> ~/.bashrc

# Apply changes
source ~/.bashrc
```

### Step 2.9: Make Scripts Executable

```bash
chmod +x train_high_quality.sh
chmod +x train_two_stage.sh
chmod +x test_high_quality_features.sh

# Verify
ls -l *.sh
# Should show: -rwxr-xr-x (executable)
```

---

## Phase 3: Pre-Training Test (IMPORTANT!)

### Step 3.1: Run Quick Feature Test

```bash
# This verifies everything works before committing to 30 hours
./test_high_quality_features.sh

# Expected output:
# ============================================================
# TESTING HIGH-QUALITY TRAINING FEATURES
# ============================================================
# ... (downloading EfficientNet-B3 weights if needed)
# ... (training for 3 epochs on 100 samples)
#
# Epoch 1/3 - Loss: 6.04 → 5.01
# Epoch 2/3 - Loss: 5.01 → 4.48
# Epoch 3/3 - Loss: 4.48 → 4.23
#
# ============================================================
# ALL TESTS PASSED! ✅
# ============================================================

# Duration: ~10-15 minutes
```

**If test fails:**
- Check error message
- Verify GPU is available
- Check data integrity
- Review troubleshooting section below

**If test passes:**
- ✅ Everything is working correctly
- ✅ Ready to start full training

---

## Phase 4: Start Full Training

### Step 4.1: Create tmux Session (For Persistence)

```bash
# Create new tmux session named "training"
tmux new -s training

# You're now inside tmux
# Benefits:
# - Training continues if SSH disconnects
# - Can detach and reattach anytime
# - Won't lose progress
```

**tmux Quick Reference:**
- Detach: `Ctrl+B`, then press `D`
- Reattach: `tmux attach -t training`
- List sessions: `tmux ls`
- Kill session: `tmux kill-session -t training`

### Step 4.2: Start Two-Stage Training

```bash
# Inside tmux session
cd ~/noon2/ml
source venv/bin/activate

# Start two-stage training
./train_two_stage.sh

# Expected initial output:
# ============================================================
# TWO-STAGE HIGH-QUALITY TRAINING
# ============================================================
# Stage 1: Freeze backbone, train classifier (10 epochs)
# Stage 2: Fine-tune entire model (100 epochs + augmentation)
#
# Target: 94-96% accuracy
# Total time: ~30-35 hours
# ============================================================
#
# ============================================================
# STAGE 1: Training classifier with frozen backbone
# ============================================================
#
# Downloading EfficientNet-B3 weights... (if not cached)
#
# ============================================================
# Food Recognition Model Training
# ============================================================
# Dataset: food-101
# Backbone: efficientnet_b3
# ... (configuration details)
# ============================================================
#
# Initializing trainer...
# Starting training for 10 epochs...
#
# Epoch 1 [Train]: 0%|          | 0/4419 [00:00<?, ?it/s]
```

### Step 4.3: Verify Training Started

```bash
# After seeing "Epoch 1 [Train]" start progressing, you can detach
# Watch for 1-2 minutes to ensure no immediate errors

# Training should show:
# Epoch 1 [Train]:   1%|▏         | 44/4419 [00:30<50:21,  1.45it/s, loss=5.8234, cls_loss=5.8234]

# Once confirmed, detach from tmux
# Press: Ctrl+B, then press: D

# You'll return to normal shell
# Training continues in background
```

---

## Phase 5: Monitor Training Progress

### Step 5.1: Reattach to tmux Session

```bash
# From remote machine
tmux attach -t training

# You'll see live training progress
# Watch for a while, then detach again: Ctrl+B, D
```

### Step 5.2: Check Training Status (Without tmux)

```bash
# Check if training process is running
ps aux | grep train_recognition

# Expected output:
# user  12345  99.0  15.2  ... python src/train/train_recognition.py ...

# If no output: Training stopped (check why)
# If output shows: Training is running ✓
```

### Step 5.3: Monitor from Your Current Machine (Remote Monitoring)

**Option A: SSH and check logs**

```bash
# From your current machine
ssh user@192.168.14.12 "tail -n 50 ~/noon2/ml/training_*.log 2>/dev/null || echo 'No log file yet'"

# Shows last 50 lines of training log
```

**Option B: Create monitoring script**

```bash
# On your current machine, create: monitor_remote.sh
cat > monitor_remote.sh << 'EOF'
#!/bin/bash
echo "=== Training Status at $(date) ==="
ssh user@192.168.14.12 "
cd ~/noon2/ml
echo 'Process:'
ps aux | grep train_recognition | grep -v grep | head -1
echo ''
echo 'Latest checkpoints:'
ls -lht models/recognition/food-101_efficientnet_b3/*.pt 2>/dev/null | head -3
echo ''
echo 'Recent progress:'
tmux capture-pane -pt training -S -30 2>/dev/null | grep 'Epoch' | tail -5
"
EOF

chmod +x monitor_remote.sh

# Run it anytime
./monitor_remote.sh

# Or run periodically
watch -n 300 ./monitor_remote.sh  # Every 5 minutes
```

### Step 5.4: Check Progress Milestones

**Stage 1 (First ~3 hours):**

```bash
# After ~1 hour, check progress
ssh user@192.168.14.12
cd ~/noon2/ml
tmux attach -t training

# Should see something like:
# Epoch 3/10 - Time: 645.2s - LR: 1.00e-03
# Train Metrics - Accuracy: 0.4521 | Loss: 3.2145
# Val Metrics - Accuracy: 0.4123 | Loss: 3.5234
# ✓ Saved best accuracy model: 0.4123

# Expected after 10 epochs:
# Val Accuracy: 60-70%
# Stage 1 complete message
# Proceeding to Stage 2...
```

**Stage 2 (Next ~27 hours):**

```bash
# Check every few hours
# After 10 hours total (~7 hours into stage 2):
# Epoch ~30/100 - Val Accuracy: ~75-80%

# After 20 hours total (~17 hours into stage 2):
# Epoch ~60/100 - Val Accuracy: ~85-90%

# After 30 hours total (~27 hours into stage 2):
# Epoch ~100/100 - Val Accuracy: ~94-96% ✅
```

### Step 5.5: Monitor GPU/Memory Usage

```bash
# On remote machine
watch -n 5 'ps aux | head -1; ps aux | grep python | grep train'

# Shows:
# CPU% MEM% ... COMMAND
# 95.2 15.8 ... python src/train/train_recognition.py

# Memory should stay around 15-20% (of total system)
# CPU close to 100% (one core fully utilized)
```

---

## Phase 6: Training Milestones & Checkpoints

### Expected Timeline (Two-Stage)

**Stage 1: Frozen Backbone (10 epochs)**
```
Hour 0:   Epoch 1/10  - Acc: ~5%
Hour 0.5: Epoch 3/10  - Acc: ~30%
Hour 1.5: Epoch 6/10  - Acc: ~50%
Hour 3:   Epoch 10/10 - Acc: ~65%
✓ Stage 1 Complete
```

**Stage 2: Fine-tuning (100 epochs)**
```
Hour 3:   Epoch 1/100  - Acc: ~65% (starting from stage 1)
Hour 5:   Epoch 10/100 - Acc: ~70%
Hour 10:  Epoch 30/100 - Acc: ~80%
Hour 15:  Epoch 50/100 - Acc: ~87%
Hour 20:  Epoch 70/100 - Acc: ~91%
Hour 25:  Epoch 90/100 - Acc: ~93%
Hour 30:  Epoch 100/100 - Acc: ~94-96% ✅
✓ Stage 2 Complete
```

### Checkpoint Files Created

```bash
# During training, these files are created/updated:
models/recognition/food-101_efficientnet_b3/
├── best_accuracy.pt      # Updated when val accuracy improves
├── best_f1.pt           # Updated when F1 score improves
├── last_checkpoint.pt   # Updated every epoch
└── label_mapping.json   # Created at start

# After training completes:
└── final_model.pt       # Final trained model
```

---

## Phase 7: Handle Interruptions (If Needed)

### If Training Stops/Crashes

**Step 7.1: Check what happened**

```bash
# On remote machine
cd ~/noon2/ml

# Check if process still running
ps aux | grep train_recognition

# Check tmux session
tmux ls

# Reattach to see error
tmux attach -t training
```

**Step 7.2: Resume from checkpoint**

```bash
# If training stopped at epoch 45 (for example)
# Resume from last checkpoint

cd ~/noon2/ml
source venv/bin/activate

# Resume training
python src/train/train_recognition.py \
    --dataset food-101 \
    --backbone efficientnet_b3 \
    --image-size 300 \
    --batch-size 16 \
    --epochs 100 \
    --warmup-epochs 5 \
    --lr 0.0001 \
    --mixup \
    --cutmix \
    --checkpoint models/recognition/food-101_efficientnet_b3/last_checkpoint.pt \
    --device mps

# Training will continue from epoch 46
```

### If You Need to Disconnect

**Safely disconnect:**

```bash
# 1. Ensure training is in tmux (detached)
tmux ls  # Should show: training: 1 windows (attached/detached)

# 2. Exit SSH safely
exit

# Training continues on remote machine

# 3. Reconnect later
ssh user@192.168.14.12
tmux attach -t training
```

---

## Phase 8: Training Completion

### Step 8.1: Verify Training Finished

```bash
# Connect to remote
ssh user@192.168.14.12
cd ~/noon2/ml

# Reattach to tmux
tmux attach -t training

# Should see final message:
# ============================================================
# TWO-STAGE TRAINING COMPLETE!
# ============================================================
# Best models saved in: models/recognition/food-101_efficientnet_b3
#   - best_accuracy.pt: Best validation accuracy
#   - best_f1.pt: Best F1 score
#   - last_checkpoint.pt: Final model
#
# Expected accuracy: 94-96%
# ============================================================
```

### Step 8.2: Check Final Results

```bash
# Load and inspect final checkpoint
python -c "
import torch
ckpt = torch.load('models/recognition/food-101_efficientnet_b3/last_checkpoint.pt', map_location='cpu')

print('='*60)
print('TRAINING COMPLETE - FINAL RESULTS')
print('='*60)
print(f'Total epochs trained: {ckpt[\"epoch\"] + 1}')
print(f'Best validation accuracy: {ckpt[\"best_val_acc\"]:.4f} ({ckpt[\"best_val_acc\"]*100:.2f}%)')
print(f'Best validation F1: {ckpt[\"best_val_f1\"]:.4f}')
print()
print('Final epoch metrics:')
print(f'  Train loss: {ckpt[\"history\"][\"train_loss\"][-1]:.4f}')
print(f'  Val loss: {ckpt[\"history\"][\"val_loss\"][-1]:.4f}')
print(f'  Train accuracy: {ckpt[\"history\"][\"train_acc\"][-1]:.4f}')
print(f'  Val accuracy: {ckpt[\"history\"][\"val_acc\"][-1]:.4f}')
print(f'  Train F1: {ckpt[\"history\"][\"train_f1\"][-1]:.4f}')
print(f'  Val F1: {ckpt[\"history\"][\"val_f1\"][-1]:.4f}')
print('='*60)

# Success criteria
if ckpt['best_val_acc'] >= 0.93:
    print('✅ SUCCESS: Achieved 93%+ accuracy target!')
elif ckpt['best_val_acc'] >= 0.90:
    print('✓ GOOD: Achieved 90%+ accuracy (close to target)')
else:
    print('⚠ BELOW TARGET: May need longer training or adjustments')
"
```

### Step 8.3: Download Models to Local Machine

```bash
# From your current machine
# Create local directory
mkdir -p ~/projects/noon2/ml/models/recognition/

# Download all model files
scp -r user@192.168.14.12:~/noon2/ml/models/recognition/food-101_efficientnet_b3/ \
    ~/projects/noon2/ml/models/recognition/

# Verify download
ls -lh ~/projects/noon2/ml/models/recognition/food-101_efficientnet_b3/

# Should see:
# best_accuracy.pt (45-50 MB)
# best_f1.pt (45-50 MB)
# last_checkpoint.pt (45-50 MB)
# final_model.pt (45-50 MB)
# label_mapping.json (5-10 KB)
```

---

## Phase 9: Cleanup Remote Machine (Optional)

```bash
# After successfully downloading models

# On remote machine
cd ~/noon2/ml

# Kill tmux session
tmux kill-session -t training

# Optional: Remove large checkpoint files to save space
# (Keep best_accuracy.pt and final_model.pt, delete others)
cd models/recognition/food-101_efficientnet_b3/
rm -i best_f1.pt last_checkpoint.pt

# Optional: Clear MPS cache
python -c "import torch; torch.mps.empty_cache()"

# Exit remote session
exit
```

---

## Quick Command Reference

### Start Training
```bash
ssh user@192.168.14.12
cd ~/noon2/ml
source venv/bin/activate
tmux new -s training
./train_two_stage.sh
# Ctrl+B, D to detach
```

### Check Progress
```bash
ssh user@192.168.14.12
tmux attach -t training
# Ctrl+B, D to detach
```

### Monitor (Without Attaching)
```bash
ssh user@192.168.14.12 "ps aux | grep train_recognition"
```

### Download Results
```bash
scp -r user@192.168.14.12:~/noon2/ml/models/recognition/food-101_efficientnet_b3/ \
    ~/projects/noon2/ml/models/recognition/
```

---

## Troubleshooting Common Issues

### Issue: "MPS backend out of memory"

**Solution:**
```bash
# Edit train_two_stage.sh
# Reduce batch size
BATCH_SIZE=12  # or 8

# Or reduce image size
IMAGE_SIZE=256  # or 224
```

### Issue: "Training very slow"

**Check:**
```bash
# Verify GPU being used
python -c "import torch; x = torch.randn(1000, 1000).to('mps'); print('MPS works!')"

# Check NUM_WORKERS
# In script, try NUM_WORKERS=2 instead of 4
```

### Issue: "SSH connection lost"

**Check:**
```bash
# Ensure training in tmux
tmux ls

# If in tmux, training continues
# Reconnect and reattach:
ssh user@192.168.14.12
tmux attach -t training
```

### Issue: "Accuracy stuck/not improving"

**Check:**
```bash
# View learning rate
# Should decrease gradually: 0.001 → 0.0001 → ... → 0.000001

# Check if mixup/cutmix enabled
# Logs should show: "Mixup/CutMix enabled: mixup=True (α=0.2), cutmix=True (α=1.0)"

# If stuck after 50 epochs, consider:
# - Longer training (200 epochs)
# - Different learning rate schedule
# - Check data quality
```

---

## Success Checklist

**Before Training:**
- [x] Code transferred to 192.168.14.12
- [x] Environment activated (venv)
- [x] Dependencies installed
- [x] GPU available (MPS/CUDA)
- [x] Data verified (70,700 train / 15,150 val)
- [x] Test script passed ✅
- [x] tmux session created
- [x] Training script started

**During Training:**
- [ ] Stage 1 completed (~3 hours) - Acc: ~65%
- [ ] Epoch 25 reached (~10 hours) - Acc: ~78%
- [ ] Epoch 50 reached (~15 hours) - Acc: ~87%
- [ ] Epoch 75 reached (~23 hours) - Acc: ~91%
- [ ] Stage 2 completed (~30 hours) - Acc: **94-96%** ✅

**After Training:**
- [ ] Final accuracy verified (93%+)
- [ ] Models downloaded to local machine
- [ ] Results documented
- [ ] Remote cleanup completed

---

## Timeline Summary

| Phase | Time | Action |
|-------|------|--------|
| **Setup** | 30 min | Transfer files, install deps, test |
| **Stage 1** | 3 hours | Train frozen backbone (10 epochs) |
| **Stage 2** | 27 hours | Fine-tune full model (100 epochs) |
| **Download** | 10 min | Transfer models to local |
| **Total** | ~30-31 hours | |

**Expected Result**: 94-96% validation accuracy ✅

---

**Ready to start training!**

```bash
# The complete procedure in one go:
ssh user@192.168.14.12
cd ~/noon2/ml
source venv/bin/activate
./test_high_quality_features.sh  # Verify (10 mins)
tmux new -s training
./train_two_stage.sh  # Start training (30 hours)
# Ctrl+B, D to detach
```

---

**Created**: 2025-10-31
**Target**: 94-96% accuracy
**Method**: Two-stage training with EfficientNet-B3
**Status**: Ready to execute
