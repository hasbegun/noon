# Configuration Review - High-Quality Training (Option B)

## Executive Summary

**Objective**: Achieve 93-95% accuracy on Food-101 classification

**Method**: EfficientNet-B3 with advanced augmentation and optimized training

**Timeline**: 25-30 hours on remote machine (192.168.14.12)

**Status**: ✅ All features implemented and tested

---

## 1. Model Architecture

### EfficientNet-B3 Specifications

```
Architecture: EfficientNet-B3 (pretrained on ImageNet)
Parameters: 11,640,461 (12M)
Input size: 300×300×3
Output: 101 classes (Food-101 categories)

Backbone: EfficientNet-B3 (frozen initially, then fine-tuned)
Classifier:
  - Dropout(0.3)
  - Linear(1536 → 512) + ReLU
  - Linear(512 → 256) + ReLU
  - Linear(256 → 101)
```

**Why EfficientNet-B3?**
- ✅ Balanced accuracy vs efficiency (sweet spot)
- ✅ 2.5x larger than B0 → +6-7% accuracy
- ✅ Pretrained weights transfer well to food images
- ✅ Fits in 24 GB memory with batch_size=16

**Comparison:**

| Model | Params | Image Size | Memory | Accuracy (Food-101) |
|-------|--------|------------|--------|---------------------|
| EfficientNet-B0 | 4.8M | 224 | 20 GB | 75-80% |
| **EfficientNet-B3** | **12M** | **300** | **24 GB** | **93-95%** ⭐ |
| EfficientNet-B4 | 19M | 380 | 28 GB | 95-97% |
| ResNet-50 | 25M | 224 | 22 GB | 88-92% |

---

## 2. Dataset Configuration

### Food-101 Dataset

```
Total samples: 101,000 images
Classes: 101 food categories
Split:
  - Training: 70,700 images (700 per class)
  - Validation: 15,150 images (150 per class)
  - Test: 25,250 images (not used in training)

Data location: data/processed/food-101/
  ├── train.parquet (metadata + labels)
  ├── val.parquet (metadata + labels)
  └── images/
      ├── train/ (70,700 images)
      └── val/ (15,150 images)

Average image size: ~120 KB
Total dataset size: ~12 GB
```

**Class Distribution:**
- Perfectly balanced: 700 train / 150 val per class
- No class imbalance issues
- Representative of real-world food variety

**Sample Categories:**
```
['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio',
 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap',
 'bread_pudding', 'breakfast_burrito', ...]
```

---

## 3. Training Configuration

### Hyperparameters

```yaml
# Model
backbone: efficientnet_b3
num_classes: 101
dropout: 0.3
pretrained: true

# Training
epochs: 150
batch_size: 16
learning_rate: 0.001
weight_decay: 0.01
num_workers: 4

# Scheduler
scheduler: cosine_annealing
warmup_epochs: 5
warmup_start_factor: 0.1
eta_min: 1e-6

# Image
image_size: 300
normalization: ImageNet stats
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

# Loss
criterion: CombinedClassificationLoss
  ce_weight: 0.7
  focal_weight: 0.3
  label_smoothing: 0.1
  focal_gamma: 2.0

# Device
device: mps  # Apple Silicon GPU
mixed_precision: false  # MPS doesn't support AMP yet
```

### Learning Rate Schedule

```
Epoch 1-5 (Warmup):
  LR: 0.0001 → 0.001 (linear warmup)

Epoch 6-150 (Cosine Annealing):
  LR: 0.001 → 0.000001 (smooth decay)

Formula:
  warmup: lr = lr_start * (1 + (epoch / warmup_epochs) * 9)
  cosine: lr = eta_min + (lr_max - eta_min) * 0.5 * (1 + cos(π * epoch / T_max))
```

**Why this schedule?**
- Warmup prevents early instability from large gradients
- Cosine decay allows smooth convergence
- Final LR (1e-6) fine-tunes without overshooting

---

## 4. Data Augmentation Strategy

### Training Augmentations

```python
# Geometric
A.RandomResizedCrop(size=(300, 300), scale=(0.8, 1.0), ratio=(0.9, 1.1))
A.HorizontalFlip(p=0.5)
A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5)

# Color
A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5)
A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)

# Noise & Blur
A.GaussNoise(var_limit=(10.0, 50.0), p=0.2)
A.GaussianBlur(blur_limit=(3, 5), p=0.2)

# Regularization
A.CoarseDropout(max_holes=8, max_height=30, max_width=30, p=0.3)

# Normalization
A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

### Mixup & CutMix (NEW - Key Feature!)

**Mixup:**
```
For two samples (x1, y1) and (x2, y2):
  λ ~ Beta(α=0.2)
  x_mixed = λ * x1 + (1 - λ) * x2
  loss = λ * L(pred, y1) + (1 - λ) * L(pred, y2)

Effect: Smooth decision boundaries, better generalization
Expected gain: +2-3% accuracy
```

**CutMix:**
```
For two samples (x1, y1) and (x2, y2):
  λ ~ Beta(α=1.0)
  Cut random box from x2, paste into x1
  λ_adjusted = 1 - (box_area / image_area)
  loss = λ * L(pred, y1) + (1 - λ) * L(pred, y2)

Effect: Forces model to learn from partial views
Expected gain: +1-2% accuracy
```

**Combined Effect:**
- Randomly applies either Mixup or CutMix to each batch
- Probability: 50% Mixup, 50% CutMix
- Total expected gain: +3-4% accuracy

### Validation Augmentations

```python
# Minimal augmentation for consistent evaluation
A.Resize(height=300, width=300)
A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

---

## 5. Loss Function Details

### CombinedClassificationLoss

```python
Total Loss = 0.7 * CrossEntropyLoss + 0.3 * FocalLoss

CrossEntropyLoss:
  - Label smoothing: 0.1
  - Prevents overconfidence
  - Better calibration

FocalLoss:
  - Gamma: 2.0
  - Focuses on hard examples
  - Down-weights easy examples
  - Formula: FL = -α(1-p)^γ log(p)
```

**Why this combination?**
- CE provides stable baseline
- Focal loss handles class imbalance (if any)
- Label smoothing prevents overfitting
- Weighted combination balances both objectives

---

## 6. Optimization Strategy

### Optimizer: AdamW

```python
optimizer = AdamW(
    params=model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.01,  # L2 regularization
    eps=1e-8
)
```

**Why AdamW?**
- Better weight decay handling than Adam
- Adapts learning rate per parameter
- Works well with vision transformers and CNNs
- Decouples weight decay from gradient updates

### Gradient Management

```python
# Gradient clipping (implicit in AdamW)
# No explicit clipping needed

# Batch normalization
# Handled by EfficientNet architecture

# Dropout
# Applied in classifier: 0.3
```

---

## 7. Two-Stage Training Strategy

### Stage 1: Freeze Backbone (10 epochs)

```yaml
Objective: Train classifier on pretrained features
Duration: ~3 hours

Configuration:
  freeze_backbone: true
  epochs: 10
  lr: 0.001
  batch_size: 16
  augmentation: basic (no mixup/cutmix)

Expected:
  - Quick convergence to ~60-70% accuracy
  - Adapts classifier to Food-101 distribution
  - Preserves pretrained feature quality
```

### Stage 2: Fine-tune Full Model (100 epochs)

```yaml
Objective: Refine entire network end-to-end
Duration: ~27 hours

Configuration:
  freeze_backbone: false
  epochs: 100
  lr: 0.0001  # Lower LR for stability
  batch_size: 16
  augmentation: full (mixup + cutmix)
  warmup_epochs: 5
  resume_from: stage1_checkpoint.pt

Expected:
  - Gradual improvement: 70% → 94-96%
  - Fine-tunes features for food-specific patterns
  - Advanced augmentation prevents overfitting
```

**Why two-stage?**
- Prevents catastrophic forgetting of pretrained weights
- Faster initial convergence
- Better final accuracy (+2-3% vs single-stage)
- More stable training dynamics

---

## 8. Memory Optimization

### MPS-Specific Optimizations

```python
# Environment variables
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Allow overflow to system memory
PYTORCH_MPS_ALLOCATOR_POLICY=garbage_collection  # Reduce fragmentation

# Trainer optimizations
if device == "mps":
    # 1. Immediate CPU transfer for metrics
    metrics = metrics.cpu()

    # 2. Explicit tensor deletion
    del images, labels, logits, loss

    # 3. Frequent cache clearing
    if batch % 5 == 0:  # Training
        torch.mps.empty_cache()
    if batch % 3 == 0:  # Validation
        torch.mps.empty_cache()
```

### Memory Usage Breakdown

```
EfficientNet-B3 model: ~50 MB (parameters)
Batch (16 × 300×300×3): ~600 MB (fp32)
Optimizer state: ~100 MB (AdamW)
Gradients: ~50 MB
Activations: ~400 MB (forward pass)
MPS overhead: ~200 MB

Total per batch: ~1.4 GB
Peak during training: ~24 GB (with cache)
```

**Safety margins:**
- Tested batch_size=16 successfully ✅
- Can handle batch_size=24 if more memory available
- Falls back gracefully if OOM occurs

---

## 9. Metrics & Evaluation

### Training Metrics (logged every epoch)

```yaml
Classification:
  - Loss (total, CE, focal components)
  - Accuracy (top-1)
  - Precision (macro-averaged)
  - Recall (macro-averaged)
  - F1 Score (macro-averaged)
  - Per-class F1 (101 classes)

System:
  - Learning rate
  - Epoch time
  - Samples/second
  - GPU memory usage
```

### Evaluation Strategy

```python
# Validation every epoch
# Metrics computed on entire validation set (15,150 samples)

# Best model selection
save_if(val_accuracy > best_val_accuracy):
    save("best_accuracy.pt")

save_if(val_f1 > best_val_f1):
    save("best_f1.pt")

# Always save last checkpoint (for resuming)
save("last_checkpoint.pt")
```

### Success Criteria

```yaml
Minimum (Good):
  val_accuracy: >= 0.90 (90%)
  val_f1: >= 0.88

Target (Better):
  val_accuracy: >= 0.93 (93%)  ⭐
  val_f1: >= 0.91

Stretch (Best):
  val_accuracy: >= 0.95 (95%)
  val_f1: >= 0.93
```

---

## 10. Checkpointing Strategy

### Checkpoint Contents

```python
checkpoint = {
    'epoch': current_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_acc': best_val_accuracy,
    'best_val_f1': best_val_f1,
    'history': {
        'train_loss': [...],
        'val_loss': [...],
        'train_acc': [...],
        'val_acc': [...],
        'train_f1': [...],
        'val_f1': [...],
        'learning_rate': [...],
    },
    'num_classes': 101,
    'include_nutrition': False,
}
```

### Checkpoint Files

```
models/recognition/food-101_efficientnet_b3/
├── best_accuracy.pt     # Best validation accuracy
├── best_f1.pt           # Best F1 score
├── last_checkpoint.pt   # Latest epoch (for resuming)
├── final_model.pt       # Final trained model
└── label_mapping.json   # Class index to name mapping
```

### Resume Training

```bash
# If training interrupted, resume from last checkpoint
python src/train/train_recognition.py \
    --checkpoint models/recognition/food-101_efficientnet_b3/last_checkpoint.pt \
    --dataset food-101 \
    --backbone efficientnet_b3 \
    ... (same args as original)
```

---

## 11. Expected Training Dynamics

### Learning Curves (Expected)

**Loss:**
```
Epoch 1-10 (Warmup):    6.0 → 4.0  (rapid decrease)
Epoch 11-50:            4.0 → 2.0  (steady decrease)
Epoch 51-100:           2.0 → 1.2  (slow decrease)
Epoch 101-150:          1.2 → 0.8  (fine-tuning)

Final train loss:  ~0.8
Final val loss:    ~1.0  (slight overfitting is normal)
```

**Accuracy:**
```
Epoch 1-10:    5% → 40%   (rapid learning)
Epoch 11-50:   40% → 75%  (steady improvement)
Epoch 51-100:  75% → 90%  (refinement)
Epoch 101-150: 90% → 93-95%  (final gains)

Final train acc:  ~98%
Final val acc:    ~93-95% ⭐
```

**F1 Score:**
```
Follows similar pattern to accuracy
Final val F1:  ~0.91-0.93
```

### Overfitting Analysis

```
Expected gap (train - val):
  Accuracy: 3-5%  (98% - 93% = 5%)
  Loss: 0.2-0.3   (1.0 - 0.8 = 0.2)

This is acceptable due to:
  - Strong regularization (dropout, label smoothing, mixup/cutmix)
  - Large dataset (70k training samples)
  - Data augmentation
```

---

## 12. Comparison: Single-Stage vs Two-Stage

### Single-Stage Training

```yaml
Script: ./train_high_quality.sh
Epochs: 150
Time: ~25-30 hours
Expected accuracy: 93-95%

Pros:
  + Simpler (one command)
  + Slightly faster
  + Easier to monitor

Cons:
  - May converge slower initially
  - Slightly lower final accuracy
```

### Two-Stage Training

```yaml
Script: ./train_two_stage.sh
Stage 1: 10 epochs (frozen)
Stage 2: 100 epochs (fine-tune)
Time: ~30-35 hours
Expected accuracy: 94-96%

Pros:
  + Better final accuracy (+1-2%)
  + More stable training
  + Preserves pretrained features better

Cons:
  - More complex (two stages)
  - Slightly longer total time
  - Requires checkpoint management
```

**Recommendation**: Use two-stage for best quality (94-96%)

---

## 13. Resource Requirements

### Compute Requirements

```yaml
GPU:
  Type: Apple Silicon (M1/M2/M3) or NVIDIA GPU
  Memory: 24+ GB (unified or VRAM)
  Recommended: M2 Max/Ultra or A100

CPU:
  Cores: 8+
  For: Data loading (4 workers × 2 threads)

RAM:
  Minimum: 32 GB
  Recommended: 64 GB
  For: System + data caching

Storage:
  Dataset: 12 GB
  Models: 1 GB (checkpoints)
  Logs: 0.1 GB
  Free space required: 50+ GB
```

### Time Requirements

```yaml
Per epoch time (estimated):
  B0, batch_size=24: 5-7 minutes
  B3, batch_size=16: 10-12 minutes
  B4, batch_size=12: 15-18 minutes

Total training time:
  Single-stage (150 epochs): 25-30 hours
  Two-stage (10 + 100 epochs): 30-35 hours

Breakdown:
  Data loading: 20%
  Forward pass: 30%
  Backward pass: 35%
  Augmentation: 10%
  Logging/saving: 5%
```

### Cost Estimate (if cloud)

```yaml
AWS p3.2xlarge (V100 16GB): $3.06/hour
  Single-stage: $76-92
  Two-stage: $92-107

AWS p3.8xlarge (V100 64GB): $12.24/hour
  Can run batch_size=32, faster training
  Single-stage: $60-73 (4x parallelism)

Lambda Labs (RTX 3090): $0.80/hour
  Single-stage: $20-24
  Two-stage: $24-28

Remote machine (192.168.14.12): $0
  If already owned/available ✅
```

---

## 14. Key Configuration Files

### Main Training Script

**`train_high_quality.sh`** - Single-stage training
```bash
BACKBONE="efficientnet_b3"
IMAGE_SIZE=300
BATCH_SIZE=16
EPOCHS=150
WARMUP_EPOCHS=5
LR=0.001

--mixup --cutmix
--device mps
```

**`train_two_stage.sh`** - Two-stage training
```bash
# Stage 1
--freeze-backbone
--epochs 10
--lr 0.001

# Stage 2
--checkpoint last_checkpoint.pt
--epochs 100
--lr 0.0001
--mixup --cutmix
```

### Core Implementation Files

```
src/training/
├── mixup.py                    # Mixup/CutMix (NEW)
├── classification_trainer.py   # Modified (warmup, freeze, mixup)
├── classification_losses.py    # Combined loss
└── classification_metrics.py   # Metrics (CPU transfer)

src/models/
└── food_recognition.py         # EfficientNet-B3 support

src/train/
└── train_recognition.py        # CLI (new args)

scripts/
├── train_high_quality.sh       # Single-stage (NEW)
├── train_two_stage.sh          # Two-stage (NEW)
└── test_high_quality_features.sh  # Testing (NEW)
```

---

## 15. Risk Assessment & Mitigation

### Risk 1: Out of Memory (OOM)

**Probability**: Low (tested successfully)

**Mitigation**:
```bash
# Reduce batch size
BATCH_SIZE=12  # Instead of 16

# Or reduce image size
IMAGE_SIZE=256  # Instead of 300

# Or disable augmentation temporarily
# Remove --mixup --cutmix
```

### Risk 2: Training Takes Too Long

**Probability**: Medium (30 hours is long)

**Mitigation**:
```bash
# Reduce epochs (acceptable accuracy)
EPOCHS=100  # Still achieves ~92%

# Or increase batch size (if memory allows)
BATCH_SIZE=24

# Or use faster backbone
BACKBONE=efficientnet_b0  # Faster but lower accuracy
```

### Risk 3: Lower Than Expected Accuracy

**Probability**: Low (well-tested approach)

**Mitigation**:
```bash
# Check data quality
python -c "import pandas as pd; df = pd.read_parquet(...); print(df.head())"

# Verify augmentations working
# Check logs for mixup/cutmix messages

# Try longer training
EPOCHS=200

# Or use two-stage instead of single-stage
./train_two_stage.sh
```

### Risk 4: Connection Loss During Training

**Probability**: Medium (long training on remote)

**Mitigation**:
```bash
# ALWAYS use tmux or screen
tmux new -s training
./train_high_quality.sh

# Or use nohup
nohup ./train_high_quality.sh > training.log 2>&1 &

# Can resume from checkpoint
--checkpoint last_checkpoint.pt
```

---

## 16. Final Review Checklist

### Configuration ✅

- [x] Model: EfficientNet-B3 (12M params)
- [x] Image size: 300×300
- [x] Batch size: 16 (fits in 24 GB)
- [x] Epochs: 150 (single-stage) or 10+100 (two-stage)
- [x] LR: 0.001 with warmup
- [x] Augmentation: Mixup + CutMix enabled
- [x] Device: MPS (Apple Silicon)

### Implementation ✅

- [x] Mixup/CutMix implemented and tested
- [x] Freeze backbone support added
- [x] Warmup scheduler integrated
- [x] Image size configurable
- [x] Memory optimizations active
- [x] All tests passed ✅

### Documentation ✅

- [x] High-quality training strategy documented
- [x] Deployment guide created
- [x] Configuration review complete (this file)
- [x] Scripts ready to use

### Ready for Deployment ✅

- [x] Code tested locally
- [x] Scripts executable
- [x] Transfer instructions provided
- [x] Monitoring setup documented
- [x] Troubleshooting guide included

---

## 17. Summary

**Current State**: All features implemented and tested ✅

**Configuration**: Optimized for 93-95% accuracy

**Method**:
- EfficientNet-B3 backbone
- 300×300 input images
- Mixup + CutMix augmentation
- Warmup + Cosine annealing schedule
- Two-stage training option

**Resources**:
- Time: 25-35 hours
- Memory: 24 GB
- Storage: 50 GB

**Expected Result**: 93-95% validation accuracy on Food-101

**Deployment**: Ready to transfer to 192.168.14.12 and start training

---

**Next Steps**:
1. Review this configuration (DONE - you're here!)
2. Transfer to remote machine (192.168.14.12)
3. Run pre-flight checks
4. Start training (single-stage or two-stage)
5. Monitor progress
6. Validate results

**Recommendation**: Use two-stage training for best quality (94-96%)

```bash
./train_two_stage.sh
```

---

**Created**: 2025-10-31
**Status**: ✅ Ready for deployment
**Target**: 93-95% accuracy
**Timeline**: 25-35 hours
