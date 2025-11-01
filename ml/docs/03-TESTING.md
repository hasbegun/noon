# Testing & Evaluation Guide

Complete guide for testing and evaluating trained food recognition models.

> 📖 **See also**:
> - [MODEL_TESTING_PLAN.md](../MODEL_TESTING_PLAN.md) - Comprehensive 10-test plan
> - [TESTING_QUICKSTART.md](../TESTING_QUICKSTART.md) - Quick start guide

---

## Quick Start (5 minutes)

Test your trained model:

```bash
bash scripts/test_model_quality.sh \
    models/recognition/food-101_efficientnet_b3/best_accuracy.pt
```

**Output**:
```
=================================================================
BASIC PERFORMANCE METRICS
=================================================================
Test Set Size: 25,250 images
Number of Classes: 101

Overall Metrics:
  Accuracy:         91.2%
  Top-5 Accuracy:   98.1%

Quality Assessment:
  ✅ Accuracy: GOOD (≥90%)
  ✅ Top-5 Accuracy: GOOD (≥97%)
  ✅ F1 Score: EXCELLENT (≥91%)
=================================================================
```

---

## What Gets Tested

### Automatic Tests

| Metric | What It Measures |
|--------|------------------|
| **Accuracy** | % of correct predictions |
| **Top-5 Accuracy** | % where correct answer is in top 5 |
| **Precision** | How accurate are positive predictions |
| **Recall** | How many positives were found |
| **F1 Score** | Balance between precision and recall |

---

## Success Criteria

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| **Accuracy** | 85% | 90% | 93% |
| **Top-5 Accuracy** | 95% | 97% | 99% |
| **F1 Score** | 83% | 88% | 91% |

### Decision Matrix

- **< 85%**: ❌ Retrain with better hyperparameters
- **85-90%**: ⚠️ Acceptable for internal use
- **90-93%**: ✅ Production-ready
- **> 93%**: 🎉 Excellent!

---

## Comprehensive Test Plans

### Test Plan 1: Basic Metrics (5 min) 🔴 CRITICAL

```bash
python src/evaluation/test_basic_metrics.py \
    --model models/recognition/.../best_accuracy.pt \
    --dataset food-101 \
    --device mps
```

**Measures**: Accuracy, top-5 accuracy, precision, recall, F1

---

### Test Plan 2-10: Advanced Testing

See [MODEL_TESTING_PLAN.md](../MODEL_TESTING_PLAN.md) for:

- **Test 2**: Per-class analysis (which foods perform well/poorly?)
- **Test 3**: Confusion matrix (what foods get confused?)
- **Test 4**: Confidence calibration (is model confidence accurate?)
- **Test 5**: Robustness testing (blur, noise, lighting)
- **Test 6**: Real-world testing (your own food photos)
- **Test 7**: Edge cases (ambiguous foods)
- **Test 8**: Cross-dataset validation (generalization)
- **Test 9**: Performance benchmarking (speed, memory)
- **Test 10**: A/B comparison (compare different models)

---

## Testing Workflow

### Phase 1: Essential (30 min)

Run after every training:

```bash
# 1. Basic metrics
python src/evaluation/test_basic_metrics.py --model <path>

# 2. Per-class analysis (TODO: implement)
python src/evaluation/test_per_class.py --model <path>

# 3. Confusion matrix (TODO: implement)
python src/evaluation/test_confusion_matrix.py --model <path>
```

**Decision**: If accuracy > 90% → Deploy!

---

### Phase 2: Quality Assurance (1-2 hours)

Before production deployment:

```bash
# 4. Confidence calibration (TODO)
python src/evaluation/test_confidence_calibration.py --model <path>

# 5. Robustness testing (TODO)
python src/evaluation/test_robustness.py --model <path>

# 9. Performance benchmark (TODO)
python src/evaluation/test_performance.py --model <path>
```

---

### Phase 3: Real-World Validation

Test on actual photos:

```bash
# 6. Real-world testing (TODO)
python src/evaluation/test_real_world.py \
    --model <path> \
    --test-dir data/real_world_photos
```

---

## Understanding Metrics

### 1. Accuracy

**What**: Percentage of correct predictions

**Example**:
- Model predicts 9,120 out of 10,000 images correctly
- Accuracy = 91.2%

**When high is bad**: For imbalanced datasets (rare classes ignored)

---

### 2. Top-5 Accuracy

**What**: Percentage where correct answer is in top 5 predictions

**Example**:
- Image is "pizza"
- Model predicts: [pizza, burger, pasta, salad, fries]
- Correct! (pizza in top 5)

**Use case**: User interfaces that show top 5 options

---

### 3. Precision

**What**: Of all "pizza" predictions, how many are actually pizza?

**Formula**: True Positives / (True Positives + False Positives)

**High precision**: Few false alarms

---

### 4. Recall

**What**: Of all actual pizzas, how many did we find?

**Formula**: True Positives / (True Positives + False Negatives)

**High recall**: Few missed detections

---

### 5. F1 Score

**What**: Harmonic mean of precision and recall

**Formula**: 2 × (Precision × Recall) / (Precision + Recall)

**Best for**: Overall model quality

---

## Results Location

```
results/quality_tests/
└── food-101_efficientnet_b3_20251104_143052/
    ├── basic_metrics.json      # Detailed metrics
    └── test_log.txt            # Full output
```

### Reading Results

```python
import json

with open('results/quality_tests/.../basic_metrics.json') as f:
    results = json.load(f)

print(f"Accuracy: {results['overall']['accuracy']:.1%}")
print(f"Top-5: {results['overall']['top5_accuracy']:.1%}")
print(f"F1: {results['macro_averaged']['f1']:.1%}")
```

---

## Common Scenarios

### Scenario 1: Just Finished Training

```bash
# Quick check
bash scripts/test_model_quality.sh <model_path>

# If accuracy > 90% → Success!
# If accuracy < 90% → Check per-class analysis
```

---

### Scenario 2: Compare Two Models

```bash
# Test model A
bash scripts/test_model_quality.sh models/.../model_a.pt

# Test model B
bash scripts/test_model_quality.sh models/.../model_b.pt

# Compare results in results/quality_tests/
```

---

### Scenario 3: Test on Different Dataset

```bash
python src/evaluation/test_basic_metrics.py \
    --model models/recognition/.../best_accuracy.pt \
    --dataset combined \
    --split test
```

---

## Troubleshooting Tests

### Error: "Label mapping not found"

```bash
# Label mapping must exist in model directory
ls models/recognition/.../
# Should see: best_accuracy.pt, label_mapping.json

# Fix: Copy from another model or retrain
```

---

### Error: "Data file not found"

```bash
# Preprocess data first
python src/train/preprocess_data.py --dataset food-101
```

---

### Low Accuracy (<85%)

**Check**:
1. Did training complete? (150 epochs?)
2. Using correct model? (EfficientNet-B3?)
3. Using correct dataset split? (--split val or test)

**Solution**: Retrain with better settings (see [02-TRAINING.md](02-TRAINING.md))

---

## Best Practices

### 1. Always Test After Training

```bash
# Add to training script
python src/train/train_recognition.py ... && \
bash scripts/test_model_quality.sh <model>
```

---

### 2. Test on Multiple Splits

- **Validation**: During development
- **Test**: Final evaluation

```bash
# Validation
python src/evaluation/test_basic_metrics.py --split val --model <path>

# Test (final)
python src/evaluation/test_basic_metrics.py --split test --model <path>
```

---

### 3. Track Metrics Over Time

| Date | Model | Accuracy | Top-5 | F1 | Notes |
|------|-------|----------|-------|----|----|
| 2025-11-01 | B3 | 91.2% | 98.1% | 90.9% | With mixup/cutmix |
| 2025-11-02 | B4 | 92.8% | 98.7% | 92.1% | Longer training |

---

## Next Steps

After testing:

1. **If accuracy ≥ 90%**: Deploy! See [06-DEPLOYMENT.md](06-DEPLOYMENT.md)
2. **If accuracy < 90%**: Improve training, see [02-TRAINING.md](02-TRAINING.md)
3. **For detailed analysis**: Implement additional test plans from [MODEL_TESTING_PLAN.md](../MODEL_TESTING_PLAN.md)

---

## Additional Resources

- **[MODEL_TESTING_PLAN.md](../MODEL_TESTING_PLAN.md)** - All 10 test plans
- **[TESTING_QUICKSTART.md](../TESTING_QUICKSTART.md)** - Quick reference
- **[02-TRAINING.md](02-TRAINING.md)** - How to improve accuracy

---

**Test your models!** 📊 Start with the quick test, then expand as needed.
