# File Descriptor Leak Fix - Complete Analysis

## Root Causes Identified

### 1. **DataLoader Worker Issue** (Primary)
```
RuntimeError: Too many open files. Communication with the workers is no longer possible.
```

**Problem**: Multiple DataLoader workers (num_workers=4) each opening files
- Each worker process opens image files
- With 70,700 images and 4 workers, thousands of file descriptors
- macOS default limit: 256 open files per process
- File descriptors not released quickly enough

**Solutions Applied**:
1. Reduce num_workers to 0 or 1 for macOS
2. Set multiprocessing sharing strategy
3. Increase system file limit

### 2. **Parquet File Not Explicitly Closed**
```python
# In classification_dataset.py line 47
self.df = pd.read_parquet(data_file)  # File opened but not explicitly closed
```

**Problem**: Parquet file handle may persist
**Solution**: Use context manager

### 3. **Image Files Not Explicitly Closed**
```python
# In classification_dataset.py line 103
image = cv2.imread(str(image_path))  # cv2 should close, but not guaranteed
```

**Problem**: cv2.imread may leave file descriptors open under high load
**Solution**: Add explicit cleanup

### 4. **No Resume Functionality**
- Training crashes → Must start from scratch
- Wastes hours of compute
- No checkpoint reuse

---

## Fixes Applied

### Fix #1: DataLoader Configuration

**File**: `src/train/train_recognition.py`

```python
# BEFORE
num_workers=4  # Too many for macOS

# AFTER
import platform

# Detect platform
is_macos = platform.system() == 'Darwin'

# Set num_workers based on platform
num_workers = 0 if is_macos else min(args.num_workers, 2)

# Set multiprocessing strategy (before creating DataLoader)
if is_macos:
    torch.multiprocessing.set_sharing_strategy('file_system')
```

### Fix #2: Explicit Parquet File Closing

**File**: `src/data_process/classification_dataset.py`

```python
# BEFORE
def __init__(self, data_file, ...):
    self.df = pd.read_parquet(data_file)

# AFTER
def __init__(self, data_file, ...):
    # Load parquet and ensure file is closed
    with open(data_file, 'rb') as f:
        self.df = pd.read_parquet(f)
    # File is now definitely closed
```

### Fix #3: Image File Cleanup

**File**: `src/data_process/classification_dataset.py`

```python
# BEFORE
def __getitem__(self, idx):
    image = cv2.imread(str(image_path))
    # ... rest of code

# AFTER
def __getitem__(self, idx):
    image_path_str = str(image_path)
    image = cv2.imread(image_path_str)

    # Ensure image is loaded as numpy array (not file handle)
    if image is not None:
        image = np.array(image, copy=True)  # Force copy, release any handles

    # ... rest of code
```

### Fix #4: Resume Training Functionality

**File**: `src/train/train_recognition.py`

```python
# Add auto-resume logic
def main():
    args = parse_args()

    # Check for existing checkpoint if not specified
    if args.checkpoint is None:
        checkpoint_dir = config.models_root / "recognition" / f"{args.dataset}_{args.backbone}"
        last_checkpoint = checkpoint_dir / "last_checkpoint.pt"

        if last_checkpoint.exists():
            logger.warning("="*60)
            logger.warning("FOUND EXISTING CHECKPOINT")
            logger.warning(f"  Path: {last_checkpoint}")
            logger.warning("  Do you want to resume? (auto-resuming in 5 seconds)")
            logger.warning("  Press Ctrl+C to cancel and start fresh")
            logger.warning("="*60)

            import time
            try:
                time.sleep(5)
                args.checkpoint = last_checkpoint
                logger.info(f"Resuming from: {last_checkpoint}")
            except KeyboardInterrupt:
                logger.info("Cancelled resume. Starting fresh training.")
```

### Fix #5: System-Level File Limit

**File**: `train_high_quality.sh` and `train_two_stage.sh`

```bash
#!/bin/bash

# Increase file descriptor limit (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS - Setting file descriptor limits"

    # Soft limit
    ulimit -n 4096

    # Show current limits
    echo "File descriptor limits:"
    echo "  Soft limit: $(ulimit -n)"
    echo "  Hard limit: $(ulimit -Hn)"
fi

# Set PyTorch multiprocessing strategy
export PYTORCH_MULTIPROCESSING_STRATEGY=file_system

# ... rest of script
```

### Fix #6: Explicit Cleanup in Trainer

**File**: `src/training/classification_trainer.py`

```python
def train_epoch(self):
    # ... existing code

    for batch in pbar:
        # ... training code

        # AFTER training step - explicit cleanup
        if self.device == "mps":
            # Delete intermediate tensors
            del images, labels, class_logits, loss
            if self.include_nutrition:
                del nutrition_pred, nutrition_target

            # Clear MPS cache
            if num_batches % 5 == 0:
                torch.mps.empty_cache()

            # Force garbage collection periodically
            if num_batches % 20 == 0:
                import gc
                gc.collect()
```

---

## Implementation Priority

### CRITICAL (Must do immediately)
1. ✅ Fix DataLoader num_workers (0 for macOS)
2. ✅ Add multiprocessing strategy
3. ✅ Add system file limit increase
4. ✅ Add auto-resume functionality

### IMPORTANT (Should do)
5. ✅ Explicit parquet file closing
6. ✅ Force numpy array copy for images
7. ✅ Add periodic garbage collection

### NICE TO HAVE
8. Monitor file descriptor usage
9. Add file limit checks at startup
10. Better error messages for file limit issues

---

## Testing Strategy

### Test 1: File Descriptor Limit
```bash
# Check current limit
ulimit -n

# If < 1024, increase
ulimit -n 4096

# Verify
ulimit -n  # Should show 4096
```

### Test 2: Resume Functionality
```bash
# Start training
python src/train/train_recognition.py --dataset food-101 --epochs 10 --device mps

# Kill it after epoch 2 (Ctrl+C)

# Restart - should auto-resume
python src/train/train_recognition.py --dataset food-101 --epochs 10 --device mps

# Should show: "Found existing checkpoint... Resuming from epoch 2"
```

### Test 3: Worker Count
```python
# In train script, verify:
import torch
from torch.utils.data import DataLoader

print(f"Platform: {platform.system()}")
print(f"Num workers: {num_workers}")  # Should be 0 on macOS
print(f"Sharing strategy: {torch.multiprocessing.get_sharing_strategy()}")  # Should be 'file_system'
```

---

## Files Modified

1. **src/train/train_recognition.py**
   - Add platform detection
   - Set num_workers=0 for macOS
   - Set multiprocessing strategy
   - Add auto-resume logic

2. **src/data_process/classification_dataset.py**
   - Explicit parquet file closing
   - Force numpy array copy for images
   - Add file handle cleanup

3. **src/training/classification_trainer.py**
   - Add periodic garbage collection
   - More aggressive memory cleanup

4. **train_high_quality.sh**
   - Add ulimit command
   - Set environment variable

5. **train_two_stage.sh**
   - Add ulimit command
   - Set environment variable

---

## Emergency Recovery

If training crashes again:

### Step 1: Check File Limits
```bash
# Current limit
ulimit -n

# Increase (temporary)
ulimit -n 10240

# For permanent fix, edit ~/.zshrc or ~/.bashrc:
echo "ulimit -n 10240" >> ~/.zshrc
source ~/.zshrc
```

### Step 2: Manual Resume
```bash
# Find latest checkpoint
ls -lt models/recognition/food-101_efficientnet_b3/*.pt | head -3

# Resume from last checkpoint
python src/train/train_recognition.py \
    --checkpoint models/recognition/food-101_efficientnet_b3/last_checkpoint.pt \
    --dataset food-101 \
    --backbone efficientnet_b3 \
    --epochs 150 \
    --device mps \
    ... (same args as before)
```

### Step 3: Nuclear Option (If Still Failing)
```bash
# Disable multiprocessing entirely
python src/train/train_recognition.py \
    --num-workers 0 \  # Force single-threaded
    --batch-size 8 \   # Reduce load
    ... (rest of args)
```

---

## Root Cause Summary

| Issue | Impact | Fix | Priority |
|-------|--------|-----|----------|
| Too many DataLoader workers | **CRITICAL** | num_workers=0 on macOS | P0 |
| Default file limit too low | **HIGH** | ulimit -n 4096 | P0 |
| No resume functionality | **HIGH** | Auto-resume from checkpoint | P0 |
| Parquet file not closed | **MEDIUM** | Context manager | P1 |
| Image handles lingering | **MEDIUM** | Force numpy copy | P1 |
| No garbage collection | **LOW** | Periodic gc.collect() | P2 |

---

**Created**: 2025-10-31
**Status**: Fixes ready to apply
**Severity**: Critical (training crashes)
**Recovery**: Resume from last checkpoint
