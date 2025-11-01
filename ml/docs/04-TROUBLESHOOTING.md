# Troubleshooting Guide

Solutions to common issues and errors.

> 📖 **See also**: [FILE_DESCRIPTOR_FIX.md](../FILE_DESCRIPTOR_FIX.md) - Detailed analysis of file descriptor issues

---

## Quick Fixes

| Problem | Quick Solution |
|---------|---------------|
| Training crashes | `ulimit -n 4096` then restart |
| Out of memory | Reduce `--batch-size` to 8 or 4 |
| Low accuracy | Train longer (150+ epochs) with EfficientNet-B3 |
| Slow training | Use `--device mps` (macOS) or `cuda` (NVIDIA) |

---

## File Descriptor Leaks (macOS)

### Symptom

```
RuntimeError: Too many open files. Communication with the workers is no longer possible.
```

### Root Cause

- macOS default limit: 256 open files
- DataLoader with multiple workers opens thousands of files
- Training with 70,700 images exceeds limit

### Solution 1: Increase File Limit

```bash
# Temporary (current session)
ulimit -n 4096

# Permanent (add to ~/.zshrc or ~/.bashrc)
echo "ulimit -n 4096" >> ~/.zshrc
source ~/.zshrc
```

### Solution 2: Auto-Fixed in Code

✅ **Already fixed!** The training script automatically:
- Sets `num_workers=0` on macOS
- Uses `torch.multiprocessing.set_sharing_strategy('file_system')`
- Sets `persistent_workers=False`

### Verification

```bash
# Check current limit
ulimit -n

# Should show: 4096 or higher
```

---

## Memory Issues

### Out of Memory During Training

**Symptom**:
```
RuntimeError: MPS backend out of memory
```

**Solutions**:

#### 1. Reduce Batch Size

```bash
python src/train/train_recognition.py \
    --batch-size 8   # Instead of 16
    --device mps
```

#### 2. Use Smaller Model

```bash
--backbone efficientnet_b0  # Instead of b3
```

#### 3. Reduce Image Size

```bash
--image-size 224  # Instead of 300
```

#### 4. Enable Memory Optimization

Already enabled in code:
```python
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_MPS_ALLOCATOR_POLICY=garbage_collection
```

---

### Out of Memory During Testing

```bash
# Reduce batch size for testing
python src/evaluation/test_basic_metrics.py \
    --batch-size 16  # Instead of 32
    --model <path>
```

---

## Training Crashes

### Auto-Resume Feature

✅ **Training automatically resumes!**

When you restart training, it will:
1. Detect existing checkpoint
2. Show last epoch and best accuracy
3. Wait 10 seconds (press Ctrl+C to cancel)
4. Resume training automatically

**No action needed!** Just restart the same command.

### Manual Resume

```bash
python src/train/train_recognition.py \
    --checkpoint models/recognition/.../last_checkpoint.pt \
    --dataset food-101 \
    --epochs 150 \
    --device mps
```

---

## Low Accuracy (<85%)

### Common Causes

1. **Not enough epochs**
   - Solution: Train for 150+ epochs
   ```bash
   --epochs 150
   ```

2. **Model too small**
   - Solution: Use EfficientNet-B3
   ```bash
   --backbone efficientnet_b3
   ```

3. **No data augmentation**
   - Solution: Enable mixup/cutmix
   ```bash
   --mixup --cutmix
   ```

4. **Wrong learning rate**
   - Solution: Use default (0.001) or tune
   ```bash
   --lr 0.001
   ```

### Full Solution

```bash
python src/train/train_recognition.py \
    --dataset food-101 \
    --backbone efficientnet_b3 \
    --epochs 150 \
    --mixup --cutmix \
    --warmup-epochs 5 \
    --device mps
```

Expected: 90-93% accuracy

---

## Data Issues

### Preprocessed Data Not Found

**Error**:
```
FileNotFoundError: data/processed/train.parquet not found
```

**Solution**:
```bash
# Preprocess data first
python src/train/preprocess_data.py --dataset food-101
```

---

### Label Mapping Not Found

**Error**:
```
FileNotFoundError: label_mapping.json not found
```

**Solution**: Label mapping is automatically saved during training. If missing:

```bash
# Retrain model (label mapping will be created)
python src/train/train_recognition.py ...

# Or copy from another model
cp models/recognition/other_model/label_mapping.json \
   models/recognition/your_model/
```

---

## Device Issues

### MPS Not Available (macOS)

**Check**:
```python
import torch
print(torch.backends.mps.is_available())
# Should print: True
```

**If False**:
- macOS 12.3+ required
- Update macOS
- Reinstall PyTorch: `pip install --force-reinstall torch`

**Workaround**:
```bash
# Use CPU instead
--device cpu
```

---

### CUDA Not Available (NVIDIA)

**Check**:
```python
import torch
print(torch.cuda.is_available())
```

**If False**:
- Install CUDA toolkit
- Install cuDNN
- Reinstall PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Slow Training

### Optimize Performance

| Issue | Solution |
|-------|----------|
| Using CPU | Use `--device mps` or `cuda` |
| Small batch size | Increase to 16 or 32 |
| Slow disk | Move data to SSD |
| num_workers=0 | Expected on macOS (prevents crashes) |

### Expected Training Times

| Hardware | Model | Batch | Time/Epoch | Total (150 epochs) |
|----------|-------|-------|------------|-------------------|
| M3 Max | B3 | 16 | 6 min | 16 hours |
| M4 Pro | B3 | 16 | 5 min | 13 hours |
| RTX 4090 | B3 | 32 | 2 min | 5 hours |

---

## Import Errors

### Module Not Found

**Error**:
```
ModuleNotFoundError: No module named 'xxx'
```

**Solution**:
```bash
# Activate conda environment
conda activate noon2

# Reinstall dependencies
pip install -r requirements.txt

# Check if module installed
pip list | grep <module-name>
```

---

### Circular Import Errors

**Solution**: Usually already fixed in code. If you encounter:

```bash
# Check Python path
echo $PYTHONPATH

# Ensure you're in correct directory
cd /path/to/noon2/ml
```

---

## Common Errors

### Error: "Unrecognized arguments: --seed 123"

✅ **Fixed!** Update your code:
```bash
git pull  # Get latest code
```

The `--seed` argument is now supported.

---

### Error: "Too many open files"

✅ **Fixed!** But you need to increase system limit:

```bash
ulimit -n 4096
```

Then restart training (it will auto-resume).

---

### Error: "Checkpoint not found"

If checkpoint exists but can't be loaded:

```bash
# Check checkpoint exists
ls models/recognition/.../last_checkpoint.pt

# Load and inspect
python -c "
import torch
ckpt = torch.load('models/.../last_checkpoint.pt', map_location='cpu')
print(f'Epoch: {ckpt[\"epoch\"]}')
print(f'Best acc: {ckpt[\"best_val_acc\"]}')
"
```

---

## Performance Issues

### Training Hangs

**Causes**:
- Deadlock in DataLoader
- MPS backend issue

**Solution**:
```bash
# Reduce num_workers (already 0 on macOS)
--num-workers 0

# Or use CPU
--device cpu
```

---

### Evaluation Too Slow

```bash
# Reduce batch size
python src/evaluation/test_basic_metrics.py \
    --batch-size 16 \
    --model <path>
```

---

## Getting Help

### Before Opening Issue

1. **Check this guide** - Most issues covered here
2. **Check logs** - Read full error message
3. **Verify setup** - Run verification tests
4. **Try solutions** - Apply quick fixes above

### Opening an Issue

Include:
1. **Full error message** (not just last line)
2. **System info**: macOS version, Python version, PyTorch version
3. **Command used**: Full command that caused error
4. **What you tried**: Solutions already attempted

### Get System Info

```bash
python -c "
import sys
import torch
import platform

print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'Platform: {platform.platform()}')
print(f'MPS: {torch.backends.mps.is_available()}')
print(f'CUDA: {torch.cuda.is_available()}')
"
```

---

## Additional Resources

- **[FILE_DESCRIPTOR_FIX.md](../FILE_DESCRIPTOR_FIX.md)** - Detailed file descriptor analysis
- **[02-TRAINING.md](02-TRAINING.md)** - Training best practices
- **[03-TESTING.md](03-TESTING.md)** - Testing guide

---

**Still stuck?** Check the detailed documentation or open an issue with full context.
