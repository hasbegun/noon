# Datasets and Incremental Training Guide

## Available Datasets

The food recognition system supports **3 dataset configurations**:

### 1. **Food-101** (Default)
- **Description**: Large-scale food classification dataset
- **Classes**: 101 food categories
- **Categories**: International dishes (pizza, sushi, tacos, etc.)
- **Use case**: General food recognition
- **Sample categories**:
  - `apple_pie`, `baby_back_ribs`, `baklava`, `beef_carpaccio`
  - `pizza`, `sushi`, `tacos`, `hamburger`, `ice_cream`
  - ... (see `src/data_process/food_labels.py` for full list)

**Training command**:
```bash
python src/train/train_recognition.py \
    --dataset food-101 \
    --backbone efficientnet_b3 \
    --epochs 150 \
    --device mps
```

---

### 2. **Nutrition5k**
- **Description**: Food dataset with nutrition information
- **Classes**: 18 food categories (broader categories)
- **Categories**: Generic food types (rice, chicken, vegetables, etc.)
- **Use case**: Nutrition-aware food recognition
- **Sample categories**:
  - `rice`, `chicken`, `beef`, `pork`, `fish`
  - `vegetables`, `salad`, `pasta`, `bread`, `soup`
  - `sandwich`, `burger`, `pizza`, `fries`, `dessert`
  - `fruit`, `mixed_dish`, `other`

**Training command**:
```bash
python src/train/train_recognition.py \
    --dataset nutrition5k \
    --backbone efficientnet_b3 \
    --epochs 100 \
    --device mps
```

**With nutrition prediction**:
```bash
python src/train/train_recognition.py \
    --dataset nutrition5k \
    --with-nutrition \
    --backbone efficientnet_b3 \
    --epochs 100 \
    --device mps
```

---

### 3. **Combined** (For Incremental Training)
- **Description**: Merged dataset with all categories from both datasets
- **Classes**: **115 food categories** (101 + 18, with overlap removed)
- **Use case**: **Incremental training across multiple datasets**
- **Benefit**: Train on one dataset, then add data from another

**Training command**:
```bash
python src/train/train_recognition.py \
    --dataset combined \
    --backbone efficientnet_b3 \
    --epochs 150 \
    --device mps
```

---

## Incremental Training Strategy

### ✅ Recommended Approach: Use "Combined" Dataset

To enable incremental training where you can add data from different datasets over time:

#### **Step 1: Initial Training on Food-101 (Using Combined Mode)**

```bash
# Train on food-101 data first, but use 'combined' label space
python src/train/train_recognition.py \
    --dataset combined \           # Use combined label space (115 classes)
    --backbone efficientnet_b3 \
    --epochs 150 \
    --batch-size 16 \
    --seed 42 \                    # For reproducibility
    --device mps
```

**What happens**:
- Model learns 115 classes (all possible categories)
- Only sees Food-101 samples during training
- Other classes will have no training data yet (that's okay!)

#### **Step 2: Add Nutrition5k Data (Incremental)**

After Step 1 completes, you can continue training with nutrition5k data:

```bash
# Resume training and add nutrition5k data
python src/train/train_recognition.py \
    --dataset combined \                    # Still use combined label space
    --backbone efficientnet_b3 \
    --checkpoint models/recognition/combined_efficientnet_b3/last_checkpoint.pt \
    --epochs 50 \                          # Additional epochs with new data
    --batch-size 16 \
    --lr 0.0001 \                          # Lower learning rate for fine-tuning
    --seed 42 \
    --device mps
```

**What happens**:
- Model loads previous weights (trained on Food-101)
- Continues training with Nutrition5k samples added
- Learns to recognize new categories from Nutrition5k
- Retains knowledge from Food-101 (transfer learning)

#### **Step 3: Continue Adding More Data**

You can keep adding data iteratively:

```bash
# Add more data sources
python src/train/train_recognition.py \
    --dataset combined \
    --checkpoint models/recognition/combined_efficientnet_b3/last_checkpoint.pt \
    --epochs 25 \
    --lr 0.00005 \                         # Even lower LR for stability
    --device mps
```

---

### ⚠️ Why Not Mix Datasets Directly?

**Problem with dataset-specific modes**:
- `--dataset food-101`: Model has 101 output classes
- `--dataset nutrition5k`: Model has 18 output classes

**These are incompatible!** You cannot resume a food-101 checkpoint for nutrition5k training because the classifier head has different dimensions.

**Solution**: Use `--dataset combined` from the start, which creates a unified label space.

---

## Dataset Comparison

| Dataset | Classes | Categories Type | Nutrition Data | Best For |
|---------|---------|----------------|----------------|----------|
| **food-101** | 101 | Specific dishes | ❌ No | High accuracy on specific foods |
| **nutrition5k** | 18 | Generic food types | ✅ Yes | Nutrition prediction |
| **combined** | 115 | Both | ✅ Partial | Incremental training |

---

## Random Seed for Reproducibility

### What is `--seed` and Why Use It?

The `--seed` argument sets the random number generator seed for:
- **PyTorch** (model weight initialization)
- **NumPy** (data augmentation randomness)
- **Python random** (data shuffling)
- **CUDA** (GPU operations, if applicable)

### Why You Need It

1. **Reproducibility**: Get identical results across multiple runs
2. **Ensemble Training**: Train multiple models with different seeds for diversity
3. **Debugging**: Isolate bugs by removing randomness
4. **Fair Comparison**: Compare different hyperparameters with same initialization

### Usage Examples

#### Single Model Training (Any Seed)
```bash
python src/train/train_recognition.py \
    --dataset food-101 \
    --seed 42 \
    --device mps
```

#### Ensemble Training (Different Seeds)

Train 3 models with different seeds for ensemble:

```bash
# Model 1
python src/train/train_recognition.py \
    --dataset food-101 \
    --backbone efficientnet_b3 \
    --seed 42 \
    --epochs 150 \
    --device mps

# Model 2
python src/train/train_recognition.py \
    --dataset food-101 \
    --backbone efficientnet_b3 \
    --seed 123 \          # Different seed
    --epochs 150 \
    --device mps

# Model 3
python src/train/train_recognition.py \
    --dataset food-101 \
    --backbone efficientnet_b4 \
    --seed 456 \          # Different seed
    --epochs 150 \
    --device mps
```

**Why different seeds?**
- Each model starts with different random weights
- Learns slightly different patterns
- Averaging their predictions improves accuracy (ensemble effect)
- Can improve accuracy by 2-4%

### What Gets Set?

```python
# When you use --seed 42, the code does:
torch.manual_seed(42)                  # PyTorch CPU
torch.cuda.manual_seed_all(42)         # PyTorch GPU
np.random.seed(42)                     # NumPy
random.seed(42)                        # Python random
torch.backends.cudnn.deterministic = True   # Deterministic ops
torch.backends.cudnn.benchmark = False      # Disable auto-tuning
```

### Performance Note

⚠️ Setting `--seed` makes training **deterministic** but may be **slightly slower** (5-10%) due to disabling CUDNN optimizations.

**Trade-off**:
- ✅ **Use seed**: When you need reproducibility or training ensembles
- ❌ **Skip seed**: When you just want maximum speed

---

## Complete Incremental Training Example

### Scenario: Train on Food-101, then add Nutrition5k

```bash
# ============================================================
# PHASE 1: Train on Food-101 data (using combined label space)
# ============================================================
echo "Phase 1: Training on Food-101 data..."

python src/train/train_recognition.py \
    --dataset combined \
    --backbone efficientnet_b3 \
    --image-size 300 \
    --batch-size 16 \
    --epochs 150 \
    --warmup-epochs 5 \
    --mixup \
    --cutmix \
    --seed 42 \
    --device mps

echo "✓ Phase 1 complete!"
echo ""

# ============================================================
# PHASE 2: Add Nutrition5k data (incremental)
# ============================================================
echo "Phase 2: Adding Nutrition5k data..."

python src/train/train_recognition.py \
    --dataset combined \
    --backbone efficientnet_b3 \
    --image-size 300 \
    --batch-size 16 \
    --checkpoint models/recognition/combined_efficientnet_b3/last_checkpoint.pt \
    --epochs 50 \                  # Fewer epochs for fine-tuning
    --lr 0.0001 \                 # Lower learning rate
    --mixup \
    --cutmix \
    --seed 42 \                   # Same seed for consistency
    --device mps

echo "✓ Phase 2 complete!"
echo ""

# ============================================================
# PHASE 3 (Optional): Final fine-tuning
# ============================================================
echo "Phase 3: Final fine-tuning..."

python src/train/train_recognition.py \
    --dataset combined \
    --backbone efficientnet_b3 \
    --image-size 300 \
    --batch-size 16 \
    --checkpoint models/recognition/combined_efficientnet_b3/last_checkpoint.pt \
    --epochs 25 \
    --lr 0.00005 \                # Even lower LR
    --mixup \
    --cutmix \
    --seed 42 \
    --device mps

echo "✓ All phases complete!"
echo "Model saved in: models/recognition/combined_efficientnet_b3/"
```

---

## Best Practices for Incremental Training

### 1. Always Use "Combined" Mode
```bash
--dataset combined  # NOT food-101 or nutrition5k
```

### 2. Lower Learning Rate When Adding Data
```bash
# Initial training
--lr 0.001

# Adding new data (phase 2)
--lr 0.0001    # 10x lower

# Fine-tuning (phase 3)
--lr 0.00005   # 20x lower
```

### 3. Use Same Seed for Consistency
```bash
--seed 42  # Keep same seed across phases
```

### 4. Fewer Epochs for Later Phases
```bash
# Phase 1: Full training
--epochs 150

# Phase 2: Adding data
--epochs 50

# Phase 3: Fine-tuning
--epochs 25
```

### 5. Monitor for Catastrophic Forgetting

Check validation accuracy on original dataset (Food-101) to ensure model doesn't "forget" what it learned:

```bash
# After each phase, evaluate on Food-101 test set
python src/train/test_recognition.py \
    --checkpoint models/recognition/combined_efficientnet_b3/best_accuracy.pt \
    --test-data data/food-101/test \
    --dataset food-101
```

---

## Summary

### ✅ Available Datasets
1. **food-101**: 101 classes, specific dishes
2. **nutrition5k**: 18 classes, generic types + nutrition
3. **combined**: 115 classes, for incremental training

### ✅ Incremental Training Works!
- Use `--dataset combined` from the start
- Train on Food-101 first
- Add Nutrition5k data later using `--checkpoint`
- Lower learning rate when adding new data

### ✅ Random Seed (`--seed`)
- Sets random number generator seed
- Ensures reproducibility
- **Required for ensemble training** (use different seeds)
- May reduce speed by ~5-10%

---

**Created**: 2025-11-01
**Status**: Feature implemented and documented
**Related**: HIGH_QUALITY_TRAINING_STRATEGY.md, FILE_DESCRIPTOR_FIX.md
