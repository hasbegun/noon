# Training Loss Goes to 0.0 - Root Cause and Fix

## Problem

Training loss drops to 0.0 on epoch 2, indicating the model is likely learning to output all zeros.

## Root Cause

After analyzing the code, the issue is:

### 1. **Empty Masks in Dataset** (`src/data_process/dataset.py` lines 90-95)
```python
# If no segmentation annotation, create empty mask
if row.get("has_segmentation", False) and pd.notna(row.get("annotation_path")):
    mask = self._load_mask(Path(row["annotation_path"]), image.shape[:2])
else:
    # Create empty mask  ← ALL ZEROS!
    sample["mask"] = np.zeros(image.shape[:2], dtype=np.uint8)
```

**If the dataset doesn't have segmentation annotations, ALL masks will be zeros!**

### 2. **Model Learns to Output Zeros**
- With BCEWithLogitsLoss and all-zero targets, the model quickly learns to output all zeros
- This gives minimal loss (close to 0.0)
- By epoch 2, the model has "converged" to the trivial solution

### 3. **No Regularization**
- No checks to prevent all-zero predictions
- No penalties for trivial solutions
- No data validation warnings

## Solutions

### Fix 1: Add Loss Regularization (RECOMMENDED)

Add a custom loss that penalizes all-zero predictions:

```python
# In src/training/trainer.py

class CombinedSegmentationLoss(nn.Module):
    """
    Combined loss with regularization to prevent all-zero predictions
    """
    def __init__(self, zero_penalty_weight=0.1):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.zero_penalty_weight = zero_penalty_weight

    def forward(self, predictions, targets):
        # Standard BCE loss
        bce = self.bce_loss(predictions, targets)

        # Penalty for all-zero predictions
        # Encourage the model to predict at least some foreground
        pred_mean = torch.sigmoid(predictions).mean()
        zero_penalty = torch.exp(-pred_mean * 10)  # Penalize if mean is close to 0

        # Combined loss
        total_loss = bce + self.zero_penalty_weight * zero_penalty

        return total_loss
```

Then update trainer to use this loss:
```python
self.criterion = CombinedSegmentationLoss()
```

### Fix 2: Add Dice Loss

Dice loss is better for segmentation with imbalanced classes:

```python
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)

        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )

        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, predictions, targets):
        return (self.bce_weight * self.bce(predictions, targets) +
                self.dice_weight * self.dice(predictions, targets))
```

### Fix 3: Filter Out Zero Masks

Update dataset to skip samples with all-zero masks:

```python
# In src/data_process/dataset.py __init__

# Filter valid samples - both file exists AND has non-zero masks
logger.info(f"Filtering samples with valid masks...")
def has_valid_mask(row):
    if not row.get("has_segmentation", False):
        return False
    if pd.isna(row.get("annotation_path")):
        return False
    # Could add additional check to verify mask file exists
    return True

valid_mask = self.df.apply(has_valid_mask, axis=1)
logger.info(f"Samples with segmentation: {valid_mask.sum()} / {len(self.df)}")

if valid_mask.sum() == 0:
    logger.error("NO SAMPLES WITH SEGMENTATION MASKS FOUND!")
    logger.error("Cannot train segmentation model without mask annotations")
    raise ValueError("No segmentation annotations in dataset")

self.df = self.df[valid_mask].reset_index(drop=True)
```

### Fix 4: Add Training Validation

Add checks in trainer to detect and warn about all-zero predictions:

```python
# In src/training/trainer.py train_epoch()

# After forward pass
pred_mean = torch.sigmoid(outputs).mean().item()
if pred_mean < 0.01:
    logger.warning(f"⚠️  Model predicting mostly zeros! Mean prediction: {pred_mean:.6f}")
```

## Implementation

### Step 1: Create Custom Loss (losses.py)
Create a new file `src/training/losses.py`:

```python
"""
Custom loss functions for food segmentation
"""
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)

        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class CombinedSegmentationLoss(nn.Module):
    """
    Combined loss for segmentation:
    - BCE for pixel-wise classification
    - Dice for overlap maximization
    - Focal for hard examples
    """
    def __init__(
        self,
        bce_weight=0.4,
        dice_weight=0.4,
        focal_weight=0.2,
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, predictions, targets):
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        focal_loss = self.focal(predictions, targets)

        total_loss = (
            self.bce_weight * bce_loss +
            self.dice_weight * dice_loss +
            self.focal_weight * focal_loss
        )

        return total_loss
```

### Step 2: Update Trainer

Modify `src/training/trainer.py`:

```python
# At top, add import
from training.losses import CombinedSegmentationLoss

# In __init__, replace:
# self.criterion = nn.BCEWithLogitsLoss()
# With:
self.criterion = CombinedSegmentationLoss()
logger.info("Using CombinedSegmentationLoss (BCE + Dice + Focal)")
```

### Step 3: Add Monitoring

In `train_epoch()`, add after loss calculation:

```python
# Monitor prediction statistics
with torch.no_grad():
    pred_probs = torch.sigmoid(outputs)
    pred_mean = pred_probs.mean().item()
    pred_std = pred_probs.std().item()
    target_mean = masks.float().mean().item()

    # Warn if predictions are collapsing
    if pred_mean < 0.01 and num_batches % 10 == 0:
        logger.warning(
            f"⚠️  Predictions collapsing to zeros! "
            f"pred_mean={pred_mean:.6f}, "
            f"target_mean={target_mean:.4f}"
        )
```

## Quick Fix

If you want a quick fix right now:

1. **Check your data**:
   ```bash
   cd ml
   python3 -c "
   import pandas as pd
   df = pd.read_parquet('data/processed/train.parquet')
   print(f'Total samples: {len(df)}')
   if 'has_segmentation' in df.columns:
       print(f'Has segmentation: {df[\"has_segmentation\"].sum()}')
   else:
       print('WARNING: has_segmentation column missing!')
   "
   ```

2. **If no segmentation annotations**, you need to either:
   - Add segmentation annotations to your dataset
   - Use a different training approach (e.g., weak supervision)
   - Generate pseudo-masks from SAM2 predictions

3. **If you do have annotations**, the mask loading might be broken. Check annotation paths.

## Verification

After fixing, you should see:
- Loss stays above 0.1 for first few epochs
- Loss gradually decreases over epochs
- Validation loss also decreases
- Model produces non-zero predictions

If loss still goes to 0.0, run the debug script to check masks:
```bash
cd ml
python debug_masks.py
```
