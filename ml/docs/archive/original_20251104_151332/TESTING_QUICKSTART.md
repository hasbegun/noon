# Model Testing Quick Start Guide

## 🚀 Quick Start: Test Your Model in 5 Minutes

After training completes, test your model quality with one command:

```bash
# Test your trained model
bash scripts/test_model_quality.sh models/recognition/food-101_efficientnet_b3/best_accuracy.pt
```

This runs the essential quality tests and generates a report in `results/quality_tests/`.

---

## 📋 What Gets Tested

### Automatic Tests

When you run the quick test script, it automatically measures:

✅ **Overall Accuracy** - % of correct predictions
✅ **Top-5 Accuracy** - % where correct answer is in top 5
✅ **Precision** - How accurate are positive predictions
✅ **Recall** - How many positives were found
✅ **F1 Score** - Harmonic mean of precision and recall
✅ **Macro Average** - Treats all classes equally
✅ **Weighted Average** - Weighted by class frequency

---

## 📊 Understanding Your Results

### Example Output

```
=================================================================
BASIC PERFORMANCE METRICS
=================================================================
Test Set Size: 25,250 images
Number of Classes: 101

Overall Metrics:
  Accuracy:         91.2%
  Top-5 Accuracy:   98.1%

Macro-Averaged (treats all classes equally):
  Precision:        90.8%
  Recall:           91.0%
  F1 Score:         90.9%

Weighted-Averaged (by class support):
  Precision:        91.0%
  Recall:           91.2%
  F1 Score:         91.1%
=================================================================

Quality Assessment:
  ✅ Accuracy: GOOD (≥90%)
  ✅ Top-5 Accuracy: GOOD (≥97%)
  ✅ F1 Score: EXCELLENT (≥91%)
=================================================================
```

### What Each Metric Means

#### 1. **Accuracy** (Most Important)
- **What it is**: Percentage of correct predictions
- **Good**: ≥ 90%
- **Excellent**: ≥ 93%
- **Use case**: "How often is the model correct?"

#### 2. **Top-5 Accuracy**
- **What it is**: Percentage where correct answer is in top 5 predictions
- **Good**: ≥ 97%
- **Excellent**: ≥ 99%
- **Use case**: "Can users correct if model shows top 5 options?"

#### 3. **Precision**
- **What it is**: Of all items predicted as "pizza", how many are actually pizza?
- **Formula**: True Positives / (True Positives + False Positives)
- **Use case**: "How trustworthy are the predictions?"

#### 4. **Recall**
- **What it is**: Of all actual pizza images, how many did we find?
- **Formula**: True Positives / (True Positives + False Negatives)
- **Use case**: "How many pizzas did we miss?"

#### 5. **F1 Score**
- **What it is**: Balance between precision and recall
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Use case**: "Overall model quality considering both false positives and false negatives"

#### 6. **Macro vs Weighted Average**
- **Macro**: Average of all classes (treats rare and common classes equally)
- **Weighted**: Average weighted by class frequency (emphasizes common classes)
- **Use case**: Check both to ensure model works well on all classes, not just common ones

---

## ✅ Success Criteria

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| **Accuracy** | 85% | 90% | 93% |
| **Top-5 Accuracy** | 95% | 97% | 99% |
| **F1 Score** | 83% | 88% | 91% |

### Decision Matrix

| Accuracy Range | Recommendation |
|----------------|----------------|
| **< 85%** | ❌ Retrain with more data or better hyperparameters |
| **85-90%** | ⚠️ Acceptable for internal use, but needs improvement |
| **90-93%** | ✅ Good! Ready for production with monitoring |
| **> 93%** | 🎉 Excellent! Production-ready |

---

## 🔍 Detailed Testing (Optional)

For more detailed analysis, see [MODEL_TESTING_PLAN.md](MODEL_TESTING_PLAN.md) for:

- **Test Plan 2**: Per-class analysis (which foods perform well/poorly?)
- **Test Plan 3**: Confusion matrix (what foods get confused?)
- **Test Plan 4**: Confidence calibration (is model confidence accurate?)
- **Test Plan 5**: Robustness testing (blur, noise, lighting)
- **Test Plan 6**: Real-world testing (your own food photos)
- **Test Plan 7**: Edge cases (ambiguous foods)
- **Test Plan 8**: Cross-dataset validation (generalization)
- **Test Plan 9**: Performance benchmarking (speed, memory)
- **Test Plan 10**: A/B comparison (compare different models)

---

## 🎯 Common Scenarios

### Scenario 1: Just Finished Training

```bash
# Quick test to check if training was successful
bash scripts/test_model_quality.sh \
    models/recognition/food-101_efficientnet_b3/best_accuracy.pt

# If accuracy > 90% → Success!
# If accuracy < 90% → Check per-class analysis for weak classes
```

### Scenario 2: Compare Two Models

```bash
# Test model A
bash scripts/test_model_quality.sh \
    models/recognition/food-101_efficientnet_b3/best_accuracy.pt

# Test model B
bash scripts/test_model_quality.sh \
    models/recognition/food-101_resnet50/best_accuracy.pt

# Compare the results in results/quality_tests/
```

### Scenario 3: Test on Different Dataset Split

```bash
# Test on validation set (default)
python src/evaluation/test_basic_metrics.py \
    --model models/recognition/food-101_efficientnet_b3/best_accuracy.pt \
    --dataset food-101 \
    --split val

# Test on test set (for final evaluation)
python src/evaluation/test_basic_metrics.py \
    --model models/recognition/food-101_efficientnet_b3/best_accuracy.pt \
    --dataset food-101 \
    --split test
```

### Scenario 4: Test Incrementally Trained Model

```bash
# Test combined model (trained on food-101 + nutrition5k)
bash scripts/test_model_quality.sh \
    models/recognition/combined_efficientnet_b3/best_accuracy.pt \
    combined
```

---

## 📁 Where Results Are Saved

```
results/quality_tests/
└── food-101_efficientnet_b3_20251104_143052/
    ├── basic_metrics.json      # Detailed metrics (JSON)
    └── test_log.txt            # Full test output log
```

### Reading Results JSON

```python
import json

# Load results
with open('results/quality_tests/.../basic_metrics.json', 'r') as f:
    results = json.load(f)

# Access metrics
print(f"Accuracy: {results['overall']['accuracy']:.1%}")
print(f"Top-5: {results['overall']['top5_accuracy']:.1%}")
print(f"F1: {results['macro_averaged']['f1']:.1%}")
```

---

## 🐛 Troubleshooting

### Error: "Label mapping not found"

```bash
# Make sure label_mapping.json exists in model directory
ls models/recognition/food-101_efficientnet_b3/

# Should see:
# - best_accuracy.pt
# - label_mapping.json  ← Must exist!
```

**Fix**: The label mapping is automatically saved during training. If missing, retrain or copy from another model of the same dataset.

---

### Error: "Data file not found"

```bash
# Make sure preprocessed data exists
ls data/processed/

# Should see:
# - train.parquet
# - val.parquet
# - test.parquet
```

**Fix**: Run preprocessing first:
```bash
python src/train/preprocess_data.py --dataset food-101
```

---

### Low Accuracy (<85%)

**Common causes**:
1. **Not enough training epochs**: Train longer (150+ epochs)
2. **Model too small**: Use EfficientNet-B3 instead of B0
3. **No augmentation**: Add mixup/cutmix
4. **Bad hyperparameters**: Check learning rate, batch size

**Fix**: Retrain with better settings:
```bash
python src/train/train_recognition.py \
    --dataset food-101 \
    --backbone efficientnet_b3 \
    --epochs 150 \
    --mixup --cutmix \
    --device mps
```

---

## 💡 Best Practices

### 1. Always Test After Training

```bash
# Add to end of training script
python src/train/train_recognition.py ... && \
bash scripts/test_model_quality.sh models/.../best_accuracy.pt
```

### 2. Test on Multiple Splits

- **Validation set**: During development
- **Test set**: Final evaluation before deployment

### 3. Save Test Results

```bash
# Organize results by date
RESULTS_DIR="results/quality_tests/$(date +%Y%m%d)"
mkdir -p "$RESULTS_DIR"

bash scripts/test_model_quality.sh <model> | tee "$RESULTS_DIR/test_output.txt"
```

### 4. Track Metrics Over Time

Create a spreadsheet tracking:
- Model name
- Training date
- Accuracy
- Top-5 accuracy
- F1 score
- Training time
- Notes

---

## 🔗 Related Documentation

- **[MODEL_TESTING_PLAN.md](MODEL_TESTING_PLAN.md)** - Complete 10-test plan
- **[HIGH_QUALITY_TRAINING_STRATEGY.md](HIGH_QUALITY_TRAINING_STRATEGY.md)** - How to train better models
- **[DATASETS_AND_INCREMENTAL_TRAINING.md](DATASETS_AND_INCREMENTAL_TRAINING.md)** - Available datasets

---

## 📞 Quick Reference Commands

```bash
# 1. Quick test (5 min)
bash scripts/test_model_quality.sh <model_path>

# 2. Detailed basic metrics
python src/evaluation/test_basic_metrics.py --model <model_path> --dataset food-101

# 3. Test on different split
python src/evaluation/test_basic_metrics.py --model <model_path> --split test

# 4. Save results to file
python src/evaluation/test_basic_metrics.py --model <model_path> --output results/metrics.json

# 5. Test with different device
python src/evaluation/test_basic_metrics.py --model <model_path> --device cpu
```

---

**Created**: 2025-11-04
**Status**: Ready to use
**Next**: Run `bash scripts/test_model_quality.sh <your_model>` to get started!
