# Fix Zero Masks Problem - Complete Solution

## Problem Summary

**Training loss: 0.0002, Val loss: 0.0000**
- NOT overfitting
- NOT a model problem
- **DATA PROBLEM: All masks are zeros!**

## Root Cause Analysis

### What We Found:

1. **100% of masks are all-zero** (no food regions marked)
2. **Only Food-101 dataset was preprocessed** (70,700 samples)
3. **Food-101 has NO segmentation annotations**
4. **Other datasets WITH segmentation exist but weren't used:**
   - UECFOOD100 - Has bounding box annotations ✓
   - UECFOODPIXCOMPLETE - Likely has pixel-level masks ✓
   - combinedsegmentationmodel - Suggests segmentation data ✓

### Why This Happened:

The preprocessing pipeline has code to extract segmentation from multiple datasets, but apparently only Food-101 was actually processed.

```bash
# What's in the training data now:
Datasets included:
  food-101: 70,700 samples (0 with segmentation)

# What's available but not used:
- UECFOOD100: Has bb_info.txt annotation files
- UECFOODPIXCOMPLETE: data/ directory exists
- Other segmentation datasets
```

## Solution: Re-preprocess with Segmentation Data

### Step 1: Stop Current Training

```bash
# Press Ctrl+C to stop training
# Current training is useless - model is just learning to predict zeros
```

### Step 2: Check Available Datasets

```bash
cd ml

# Check which datasets have annotations
ls data/raw/UECFOOD100/1/*.txt
ls data/raw/UECFOODPIXCOMPLETE/

# Check preprocessing code
cat src/data_process/preprocessing.py | grep "def _process"
```

### Step 3: Re-run Preprocessing

The preprocessing script should include datasets with segmentation:

```bash
cd ml
conda activate noon2

# Option A: Use Make (if available)
make preprocess

# Option B: Run preprocessing script directly
python src/train/preprocess_data.py
```

### Step 4: Verify New Data Has Segmentation

After preprocessing, check the data:

```bash
python -c "
import pandas as pd

df = pd.read_parquet('data/processed/train.parquet')

print('Datasets included:')
print(df['dataset'].value_counts())

print('\nSegmentation status:')
print(f'Samples with segmentation: {df[\"has_segmentation\"].sum()}')
print(f'Samples without segmentation: {(~df[\"has_segmentation\"]).sum()}')

print('\nSegmentation by dataset:')
print(df.groupby('dataset')['has_segmentation'].sum())
"
```

**Expected output should show samples with has_segmentation=True!**

### Step 5: Verify Masks Are Not Empty

Run the debug script:

```bash
python debug_masks.py
```

**You should now see:**
```
Training Mask Statistics:
  Partial masks: XXX (>0%) ← NOT all zeros!
  Average positive ratio: >0.00%

✓ No major issues detected
```

### Step 6: Restart Training

Only after verifying masks are not all zeros:

```bash
# Delete old checkpoints (trained on bad data)
rm models/segmentation/last_checkpoint.pt
rm models/segmentation/best_model.pt

# Start fresh training with REAL masks
caffeinate python src/train/train.py --epochs 50 --batch-size 8 --device mps
```

## Alternative: If Preprocessing Doesn't Include Segmentation Datasets

If re-preprocessing still only includes Food-101, you need to modify the preprocessing:

### Option 1: Force Include UECFOOD100

Edit `src/data_process/preprocessing.py` or create a custom script:

```python
# Ensure UECFOOD100 is processed
processor = DataPreprocessor()

# Check which datasets exist
for name in ['UECFOOD100', 'UECFOODPIXCOMPLETE', 'nutrition5k']:
    path = config.raw_data_path / name
    print(f"{name}: {'EXISTS' if path.exists() else 'NOT FOUND'}")

# Process
df = processor.process_all_datasets()
```

### Option 2: Generate Synthetic Masks

If no segmentation data is available, generate masks from:

1. **Entire image as foreground** (simple but inaccurate)
2. **Use SAM2 to generate pseudo-labels** (better)
3. **Use pretrained detector** to create masks

### Option 3: Use Different Approach

Without segmentation annotations, consider:

1. **Classification instead of segmentation**
2. **Weakly-supervised segmentation** (class activation maps)
3. **Self-supervised pre-training**

## Expected Behavior After Fix

### Before (Current - BROKEN):
```
Mask Analysis:
  All zeros: 800 (100.0%) ← PROBLEM!

Training:
  Train Loss: 0.0002 ← Too low
  Val Loss: 0.0000 ← Zero

Model: Predicts all zeros (useless)
```

### After (Fixed - WORKING):
```
Mask Analysis:
  Partial masks: 750 (93.8%) ← GOOD!
  Average positive ratio: 15.2% ← GOOD!

Training:
  Train Loss: 0.45 ← Reasonable
  Val Loss: 0.52 ← Reasonable

Model: Actually learning to segment food!
```

## How to Verify Training is Working

After restarting with proper data:

1. **Losses should be higher** (0.3-0.8 range initially)
2. **Losses should decrease over time**
3. **Val loss should be close to train loss** (not zero!)
4. **Model should output non-zero predictions**

## Debugging Commands

```bash
# Check processed data
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/train.parquet')
print(f'Total: {len(df)}')
print(f'With segmentation: {df[\"has_segmentation\"].sum()}')
print(f'Datasets: {df[\"dataset\"].unique()}')
"

# Check if masks load correctly
python debug_masks.py

# Check a specific image
python -c "
from src.data_process.dataset import FoodDataset
from pathlib import Path

dataset = FoodDataset(Path('data/processed/train.parquet'))
sample = dataset[0]
print(f'Image shape: {sample[\"image\"].shape}')
print(f'Mask shape: {sample[\"mask\"].shape}')
print(f'Mask unique values: {sample[\"mask\"].unique()}')
print(f'Positive pixels: {(sample[\"mask\"] > 0).sum().item()}')
"
```

## Summary Checklist

Before restarting training, ensure:

- [ ] Preprocessing includes datasets with segmentation
- [ ] `has_segmentation=True` for >0 samples
- [ ] Masks are not all zeros (debug_masks.py)
- [ ] annotation_path column exists and has values
- [ ] Old checkpoints deleted (trained on bad data)
- [ ] Training losses are reasonable (0.3-0.8 range)

## Don't Worry!

This is a **data pipeline issue**, not a model issue. Once you have proper segmentation annotations:
- Model will train normally
- Losses will be reasonable
- You'll see actual food segmentation!

---

**Next Step**: Re-run preprocessing to include datasets with segmentation annotations, then verify masks before restarting training.
