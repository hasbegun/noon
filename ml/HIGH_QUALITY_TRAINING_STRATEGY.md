# High-Quality Training Strategy for 95%+ Accuracy

## Goal
Achieve >95% accuracy on Food-101 classification, prioritizing quality over speed.

## 📚 Related Documentation
- **[Datasets & Incremental Training Guide](DATASETS_AND_INCREMENTAL_TRAINING.md)** - Available datasets, incremental training, and `--seed` usage
- **[File Descriptor Fix](FILE_DESCRIPTOR_FIX.md)** - Crash recovery and resume training

## Current Baseline
- Model: EfficientNet-B0 (4.8M parameters)
- Image size: 224×224
- Batch size: 24
- Epochs: 50
- **Expected accuracy: ~75-80%**

## Target
- **Accuracy: >95%**
- Time: Flexible (quality over speed)
- Memory: Work within MPS constraints

---

## Strategy Overview

### Phase 1: Foundation (Current Setup) ⏱️ ~6-8 hours
**Expected accuracy: 75-80%**

### Phase 2: Model Upgrade ⏱️ ~12-16 hours
**Expected accuracy: 82-87%**

### Phase 3: Advanced Training ⏱️ ~20-30 hours
**Expected accuracy: 88-93%**

### Phase 4: Fine-tuning & Ensemble ⏱️ ~40-60 hours total
**Expected accuracy: 95%+**

---

## Phase 1: Foundation Training (Baseline)

### Configuration
```bash
python src/train/train_recognition.py \
    --dataset food-101 \
    --backbone efficientnet_b0 \
    --batch-size 24 \
    --epochs 50 \
    --device mps
```

### Expected Results
- Time: 6-8 hours
- Accuracy: 75-80%
- **Purpose**: Establish baseline, verify pipeline works

### Action Items
1. ✅ Memory optimizations (already done)
2. ✅ Training pipeline verified
3. 🔄 Run full 50-epoch training
4. 📊 Analyze results and failure cases

---

## Phase 2: Model Architecture Upgrade

### 2.1 Larger Backbone
**EfficientNet-B0 → EfficientNet-B3**

```python
# In src/models/food_recognition.py
# EfficientNet-B0: 4.8M params, 224×224
# EfficientNet-B3: 12M params, 300×300 (optimal)
# EfficientNet-B4: 19M params, 380×380
```

**Configuration**:
```bash
python src/train/train_recognition.py \
    --dataset food-101 \
    --backbone efficientnet_b3 \
    --batch-size 16 \          # Reduced for larger model
    --epochs 100 \              # More epochs for convergence
    --image-size 300 \          # Optimal for B3
    --device mps
```

**Memory estimate**:
- EfficientNet-B3 at 300×300 with batch_size=16: ~22-25 GB
- Should fit in MPS memory with aggressive optimizations

**Expected improvement**: +5-7% accuracy (→ 82-87%)

---

### 2.2 Alternative: ResNet-50 with Deeper Classifier

```bash
python src/train/train_recognition.py \
    --dataset food-101 \
    --backbone resnet50 \
    --batch-size 20 \
    --epochs 100 \
    --device mps
```

**Expected improvement**: +4-6% accuracy (→ 80-85%)

---

## Phase 3: Advanced Training Techniques

### 3.1 Extended Training with Better Scheduling

**Key changes**:
- More epochs: 50 → 150
- Warmup: 5 epochs
- Cosine annealing with restarts
- Early stopping (patience=15)

```bash
python src/train/train_recognition.py \
    --dataset food-101 \
    --backbone efficientnet_b3 \
    --batch-size 16 \
    --epochs 150 \
    --warmup-epochs 5 \
    --scheduler cosine_restart \
    --early-stopping \
    --patience 15 \
    --device mps
```

**Expected improvement**: +2-3% accuracy

---

### 3.2 Enhanced Data Augmentation

**Current augmentations**:
- RandomResizedCrop
- HorizontalFlip
- ColorJitter
- ShiftScaleRotate
- CoarseDropout

**Add advanced augmentations**:
```python
# In classification_dataset.py
A.Compose([
    # Existing augmentations...

    # Advanced augmentations
    A.RandomBrightnessContrast(
        brightness_limit=0.3,
        contrast_limit=0.3,
        p=0.5
    ),
    A.HueSaturationValue(
        hue_shift_limit=20,
        sat_shift_limit=30,
        val_shift_limit=20,
        p=0.5
    ),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),

    # Cutout/CoarseDropout (already have)
    A.CoarseDropout(
        max_holes=8,
        max_height=32,
        max_width=32,
        fill_value=0,
        p=0.5
    ),
])
```

**Expected improvement**: +1-2% accuracy

---

### 3.3 Mixup / CutMix Data Augmentation

Implement batch-level augmentation:

```python
# New file: src/training/mixup.py
def mixup_data(x, y, alpha=0.2):
    """Mixup data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    """CutMix data augmentation"""
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).to(x.device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]

    return x, y, y[rand_index], lam
```

**Usage in trainer**:
```bash
python src/train/train_recognition.py \
    --dataset food-101 \
    --backbone efficientnet_b3 \
    --batch-size 16 \
    --epochs 150 \
    --mixup \              # Enable mixup
    --cutmix \             # Enable cutmix
    --mixup-alpha 0.2 \
    --cutmix-alpha 1.0 \
    --device mps
```

**Expected improvement**: +2-4% accuracy

---

### 3.4 Test-Time Augmentation (TTA)

Apply augmentations during inference and average predictions:

```python
# In inference_recognition.py
def predict_with_tta(self, image, num_augmentations=5):
    """Predict with test-time augmentation"""
    predictions = []

    for _ in range(num_augmentations):
        # Apply random augmentations
        augmented = self.tta_transform(image=image)['image']
        pred = self.model(augmented.unsqueeze(0))
        predictions.append(pred)

    # Average predictions
    avg_pred = torch.stack(predictions).mean(dim=0)
    return avg_pred
```

**Expected improvement**: +1-2% accuracy

---

### 3.5 Two-Stage Training (Transfer Learning)

**Stage 1: Freeze backbone** (10 epochs)
```bash
python src/train/train_recognition.py \
    --dataset food-101 \
    --backbone efficientnet_b3 \
    --batch-size 16 \
    --epochs 10 \
    --freeze-backbone \      # Only train classifier
    --lr 0.001 \
    --device mps
```

**Stage 2: Unfreeze and fine-tune** (100 epochs)
```bash
python src/train/train_recognition.py \
    --dataset food-101 \
    --backbone efficientnet_b3 \
    --batch-size 16 \
    --epochs 100 \
    --resume checkpoint.pt \  # Resume from stage 1
    --lr 0.0001 \            # Lower LR for fine-tuning
    --device mps
```

**Expected improvement**: +2-3% accuracy

---

### 3.6 Discriminative Learning Rates

Use different learning rates for different layers:

```python
# In classification_trainer.py
def get_optimizer_groups(model, base_lr, backbone_lr_ratio=0.1):
    """Create parameter groups with different learning rates"""
    backbone_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            classifier_params.append(param)

    return [
        {'params': backbone_params, 'lr': base_lr * backbone_lr_ratio},
        {'params': classifier_params, 'lr': base_lr}
    ]
```

**Expected improvement**: +1-2% accuracy

---

## Phase 4: Final Push to 95%+

### 4.1 Larger Image Size

Train with higher resolution for final model:

```bash
python src/train/train_recognition.py \
    --dataset food-101 \
    --backbone efficientnet_b3 \
    --batch-size 8 \           # Reduced for larger images
    --epochs 150 \
    --image-size 384 \         # Higher resolution
    --device mps
```

**Memory estimate**: ~25-28 GB (should fit with optimizations)

**Expected improvement**: +1-3% accuracy

---

### 4.2 Ensemble Models

Train multiple models and ensemble:

**Model 1: EfficientNet-B3**
```bash
python src/train/train_recognition.py \
    --dataset food-101 \
    --backbone efficientnet_b3 \
    --batch-size 16 \
    --epochs 150 \
    --seed 42 \                    # ✅ FIXED: --seed argument now supported!
    --device mps
```

**Model 2: EfficientNet-B4**
```bash
python src/train/train_recognition.py \
    --dataset food-101 \
    --backbone efficientnet_b4 \
    --batch-size 12 \
    --epochs 150 \
    --image-size 380 \
    --seed 123 \                   # ✅ Different seed for ensemble diversity
    --device mps
```

**Model 3: ResNet-101**
```bash
python src/train/train_recognition.py \
    --dataset food-101 \
    --backbone resnet101 \
    --batch-size 16 \
    --epochs 150 \
    --seed 456 \                   # ✅ Different seed for ensemble diversity
    --device mps
```

**Why different seeds?**
- Each model starts with different random initialization
- Creates diverse models that learn different patterns
- Averaging predictions improves ensemble accuracy by 2-4%
- See `DATASETS_AND_INCREMENTAL_TRAINING.md` for details

**Ensemble predictions**:
```python
def ensemble_predict(models, image):
    predictions = []
    for model in models:
        pred = model(image)
        predictions.append(F.softmax(pred, dim=1))

    # Average probabilities
    ensemble_pred = torch.stack(predictions).mean(dim=0)
    return ensemble_pred
```

**Expected improvement**: +2-4% accuracy

---

### 4.3 Knowledge Distillation (Optional)

Use a large teacher model to train a smaller student:

```python
# Train large teacher (EfficientNet-B4)
teacher = train_large_model()

# Distill to smaller student (EfficientNet-B3)
def distillation_loss(student_logits, teacher_logits, labels, alpha=0.7, T=3):
    """Knowledge distillation loss"""
    # Soft targets from teacher
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)

    # Hard targets from labels
    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss
```

**Expected improvement**: +1-2% accuracy for student model

---

## Complete Roadmap to 95%+

### Quick Reference

| Phase | Technique | Time | Accuracy Gain | Cumulative |
|-------|-----------|------|---------------|------------|
| **Baseline** | EfficientNet-B0, 50 epochs | 6h | - | 75-80% |
| **2.1** | Upgrade to B3, 100 epochs | +10h | +6% | 82-86% |
| **3.1** | 150 epochs, better scheduling | +6h | +2% | 84-88% |
| **3.2** | Advanced augmentation | +0h | +1.5% | 85.5-89.5% |
| **3.3** | Mixup/CutMix | +0h | +3% | 88.5-92.5% |
| **3.4** | Test-time augmentation | +0h | +1.5% | 90-94% |
| **3.5** | Two-stage training | +2h | +2% | 92-96% |
| **4.1** | Higher resolution (384) | +8h | +1.5% | 93.5-97.5% |
| **4.2** | Ensemble (3 models) | +30h | +2% | **95.5-99.5%** |

---

## Recommended Implementation Order

### Priority 1: High Impact, Low Effort
1. ✅ **Upgrade to EfficientNet-B3** (biggest single improvement)
2. ✅ **More epochs (150)** with early stopping
3. ✅ **Mixup/CutMix** augmentation

**Expected**: ~88-92% accuracy

---

### Priority 2: Medium Impact, Medium Effort
4. ✅ **Two-stage training** (freeze then unfreeze)
5. ✅ **Test-time augmentation**
6. ✅ **Enhanced data augmentation**

**Expected**: ~92-94% accuracy

---

### Priority 3: Final Push
7. ✅ **Higher resolution training** (384×384)
8. ✅ **Ensemble multiple models**

**Expected**: **95%+ accuracy** ✅

---

## Implementation Plan

### Step 1: Implement Missing Features (Priority)

**Need to add**:
1. `--image-size` argument support
2. `--warmup-epochs` support
3. `--mixup` and `--cutmix` support
4. `--freeze-backbone` support
5. Test-time augmentation in inference
6. Ensemble prediction script

---

### Step 2: Quick Wins (Start Here)

Run this first to get to ~88%:

```bash
# Train with EfficientNet-B3
python src/train/train_recognition.py \
    --dataset food-101 \
    --backbone efficientnet_b3 \
    --batch-size 16 \
    --epochs 150 \
    --device mps
```

**Time**: ~12-16 hours
**Expected**: 82-88% accuracy

---

### Step 3: Add Advanced Features

Implement mixup/cutmix, then retrain:

```bash
python src/train/train_recognition.py \
    --dataset food-101 \
    --backbone efficientnet_b3 \
    --batch-size 16 \
    --epochs 150 \
    --mixup \
    --cutmix \
    --device mps
```

**Time**: +12-16 hours
**Expected**: 88-92% accuracy

---

### Step 4: Final Training

Two-stage training + TTA:

```bash
# Stage 1: Freeze backbone
python src/train/train_recognition.py \
    --dataset food-101 \
    --backbone efficientnet_b3 \
    --batch-size 16 \
    --epochs 10 \
    --freeze-backbone \
    --device mps

# Stage 2: Fine-tune
python src/train/train_recognition.py \
    --dataset food-101 \
    --backbone efficientnet_b3 \
    --batch-size 16 \
    --epochs 100 \
    --resume last_checkpoint.pt \
    --lr 0.0001 \
    --mixup \
    --cutmix \
    --device mps
```

**Time**: +14-18 hours
**Expected**: 92-95% accuracy

---

### Step 5: Ensemble (If Needed)

Train 3 models and ensemble:

```bash
# Model 1
python src/train/train_recognition.py \
    --dataset food-101 \
    --backbone efficientnet_b3 \
    --batch-size 16 \
    --epochs 150 \
    --seed 42 \
    --output-dir models/ensemble/model1 \
    --device mps

# Model 2
python src/train/train_recognition.py \
    --dataset food-101 \
    --backbone efficientnet_b4 \
    --batch-size 12 \
    --epochs 150 \
    --seed 123 \
    --output-dir models/ensemble/model2 \
    --device mps

# Model 3
python src/train/train_recognition.py \
    --dataset food-101 \
    --backbone resnet101 \
    --batch-size 16 \
    --epochs 150 \
    --seed 456 \
    --output-dir models/ensemble/model3 \
    --device mps

# Ensemble
python src/train/ensemble_predict.py \
    --models models/ensemble/model1 models/ensemble/model2 models/ensemble/model3 \
    --test-dir data/food-101/test
```

**Time**: ~40-50 hours (can run sequentially overnight)
**Expected**: **95-98% accuracy** ✅

---

## Memory Considerations

### With Current Optimizations

| Model | Image Size | Batch Size | Memory | Feasible? |
|-------|------------|------------|--------|-----------|
| EfficientNet-B0 | 224 | 24 | 20 GB | ✅ Yes |
| EfficientNet-B3 | 300 | 16 | 24 GB | ✅ Yes |
| EfficientNet-B3 | 384 | 8 | 26 GB | ✅ Tight |
| EfficientNet-B4 | 380 | 12 | 28 GB | ⚠️ May need to reduce |
| ResNet-101 | 224 | 16 | 22 GB | ✅ Yes |

### If Memory Issues
1. Reduce batch size by 25% (16 → 12)
2. Use gradient accumulation (simulate larger batches)
3. Train overnight when system is idle

---

## Timeline Estimate

### Conservative Estimate (Sequential Training)
- Baseline (B0): 6-8 hours
- Upgrade to B3: 12-16 hours
- Advanced techniques: 14-18 hours
- Ensemble (3 models): 40-50 hours
- **Total: 72-92 hours (~3-4 days of training)**

### Accelerated (Parallel + Smart Choices)
- Skip baseline, go straight to B3: -6h
- Run ensemble models in parallel (if you have time): -20h
- Focus on single best model instead of ensemble: -40h
- **Minimum: ~20-30 hours to reach 92-95%**

---

## Success Metrics

### Minimum Viable (Good)
- Single EfficientNet-B3 model
- 150 epochs, mixup/cutmix
- **Accuracy: 88-92%**
- Time: ~16-20 hours

### Target (Better)
- EfficientNet-B3 with two-stage training
- Advanced augmentations, TTA
- **Accuracy: 92-95%**
- Time: ~25-35 hours

### Stretch Goal (Best)
- Ensemble of 3+ models
- Higher resolution, full optimization
- **Accuracy: 95-98%**
- Time: ~60-90 hours

---

## Next Steps

### Immediate Action
1. **Choose target accuracy**: 90%, 93%, or 95%+
2. **Implement priority features**: image_size support, mixup/cutmix
3. **Start first training run**: EfficientNet-B3, 150 epochs

### What Would You Like to Do?

**Option A: Quick to 90%** (1 day)
- Train EfficientNet-B3 with current features
- ~16 hours

**Option B: Solid 93%** (2-3 days)
- Add mixup/cutmix, train B3
- ~30 hours

**Option C: Target 95%+** (4-5 days)
- Full implementation, ensemble
- ~70-90 hours

---

**Created**: 2025-10-31
**Goal**: >95% accuracy on Food-101
**Status**: Strategy defined, ready to implement
**Next**: Choose approach and implement features
