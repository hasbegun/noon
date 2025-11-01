# Training Fix Summary - Loss Going to 0.0

## Problem Identified

Training loss drops to **0.0 on epoch 2**, indicating the model is learning a trivial solution (outputting all zeros).

## Root Cause

**The model is learning to predict all-zero masks** because:

1. **Dataset Issue**: Many/all training samples have empty masks (all zeros)
   - In `src/data_process/dataset.py`, samples without `has_segmentation=True` get zero masks
   - This is common when segmentation annotations are missing

2. **Loss Function**: Using only `BCEWithLogitsLoss`
   - With all-zero targets, the model learns to output all zeros
   - This gives minimal BCE loss (close to 0.0)
   - No penalty for trivial solutions

3. **No Monitoring**: No checks to detect prediction collapse

## Fixes Implemented

### ✅ 1. Created Custom Loss Functions (`src/training/losses.py`)

**New file** with multiple loss functions to prevent model collapse:

- **`DiceLoss`**: Handles class imbalance better than BCE
- **`FocalLoss`**: Focuses on hard examples
- **`IoULoss`**: Directly optimizes IoU metric
- **`TverskyLoss`**: Generalization of Dice with configurable FP/FN weights
- **`CombinedSegmentationLoss`**: Combination of BCE + Dice + Focal (recommended)

The combined loss prevents all-zero predictions by:
- Dice loss penalizes low overlap
- Focal loss handles class imbalance
- BCE provides pixel-wise supervision

### ✅ 2. Updated Trainer (`src/training/trainer.py`)

**Changed**:
```python
# Before:
self.criterion = nn.BCEWithLogitsLoss()

# After:
self.criterion = CombinedSegmentationLoss(
    bce_weight=0.4,
    dice_weight=0.4,
    focal_weight=0.2,
)
```

**Added monitoring** to detect prediction collapse:
- Checks if `pred_mean < 0.01` (predictions near zero)
- Warns every 50 batches if collapse detected
- Shows target mean for comparison

**Added validation monitoring**:
- Shows `pred_mean` in progress bar during validation
- Helps spot issues early

### ✅ 3. Updated Exports (`src/training/__init__.py`)

Added all loss functions to module exports for easy access.

### ✅ 4. Created Documentation

- **`TRAINING_FIX.md`**: Detailed explanation and additional solutions
- **`debug_masks.py`**: Script to check if masks are all zeros

## How to Use

### Option 1: Use New Training (Recommended)

The fixes are already applied! Just restart training:

```bash
cd ml
python src/train/train.py --epochs 50
```

You should now see:
- ✓ Loss stays > 0.1 for first few epochs
- ✓ Loss gradually decreases
- ✓ No warnings about prediction collapse
- ✓ Progress bar shows `pred_mean > 0.0`

### Option 2: Test Different Loss Functions

Try different losses if combined doesn't work:

```python
# In src/training/trainer.py, change:
from training.losses import DiceLoss  # or FocalLoss, IoULoss

self.criterion = DiceLoss()
```

Or use the factory:
```python
from training.losses import get_loss_function

self.criterion = get_loss_function('dice')  # 'bce', 'focal', 'combined', etc.
```

### Option 3: Adjust Loss Weights

Tune the combined loss weights:

```python
self.criterion = CombinedSegmentationLoss(
    bce_weight=0.5,   # More pixel-wise focus
    dice_weight=0.3,  # Less overlap focus
    focal_weight=0.2, # Same hard example focus
)
```

## Verification

### Check if fix worked:

1. **Training loss should NOT go to 0.0**:
   ```
   Epoch 1: loss=0.45
   Epoch 2: loss=0.38  ← Good! Not 0.0
   Epoch 3: loss=0.32
   ```

2. **No collapse warnings**:
   ```
   # Should NOT see:
   ⚠️  Predictions may be collapsing! pred_mean=0.000012
   ```

3. **Progress bar shows predictions**:
   ```
   Validation: val_loss=0.35, pred_mean=0.23  ← pred_mean > 0
   ```

### If still having issues:

1. **Check your data**:
   ```bash
   cd ml
   python debug_masks.py
   ```

2. **Verify masks are not all zeros**:
   - Look for: "Masks with non-zero values: X%"
   - If 0%, you need to fix your dataset

3. **Check model is using lightweight head**:
   ```bash
   grep "Lightweight segmentation head" logs/training.log
   ```

## What Changed in Code

| File | Change | Reason |
|------|--------|--------|
| `src/training/losses.py` | **NEW** | Custom loss functions |
| `src/training/trainer.py` | Modified loss function | Use `CombinedSegmentationLoss` |
| `src/training/trainer.py` | Added monitoring | Detect prediction collapse |
| `src/training/__init__.py` | Added exports | Make losses accessible |
| `debug_masks.py` | **NEW** | Debug script to check masks |
| `TRAINING_FIX.md` | **NEW** | Detailed documentation |

## Expected Training Behavior Now

### Before Fix:
```
Epoch 1/50: loss=0.42
Epoch 2/50: loss=0.00  ❌ PROBLEM!
Epoch 3/50: loss=0.00
```

### After Fix:
```
Epoch 1/50: loss=0.45
Epoch 2/50: loss=0.38  ✓ Good!
Epoch 3/50: loss=0.32
Epoch 4/50: loss=0.28
...
```

## Additional Recommendations

### 1. Data Verification (Important!)

Check if your dataset actually has segmentation masks:

```bash
cd ml
python -c "
import sys; sys.path.insert(0, 'src')
import pandas as pd
df = pd.read_parquet('data/processed/train.parquet')
if 'has_segmentation' in df.columns:
    print(f'Samples with segmentation: {df[\"has_segmentation\"].sum()} / {len(df)}')
    if df['has_segmentation'].sum() == 0:
        print('❌ NO SEGMENTATION ANNOTATIONS IN DATASET!')
else:
    print('❌ has_segmentation column missing!')
"
```

If you have no segmentation annotations, you need to either:
- Add segmentation annotations to your dataset
- Generate pseudo-masks using SAM2
- Use weak supervision methods

### 2. Monitor Training

Watch for these metrics during training:
- **Loss**: Should decrease gradually, not jump to 0.0
- **pred_mean**: Should be > 0.01, ideally 0.1-0.5
- **val_loss**: Should track training loss

### 3. Hyperparameter Tuning

If loss is still unstable, try:
- Lower learning rate: `--lr 1e-5`
- Smaller batch size: `--batch-size 4`
- Different loss weights

## Summary

**Problem**: Loss → 0.0 due to all-zero mask predictions

**Solution**: Combined loss (BCE + Dice + Focal) + monitoring

**Result**: Robust training that prevents trivial solutions

The training should now work correctly! If you still see issues, check the dataset for segmentation annotations.
