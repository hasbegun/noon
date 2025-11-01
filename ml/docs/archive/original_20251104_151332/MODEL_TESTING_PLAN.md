# Model Testing & Quality Evaluation Plan

## Overview

This document provides comprehensive test plans to measure the quality of your trained food recognition model. The tests are organized by **purpose** and **priority**, ranging from basic accuracy metrics to advanced real-world scenario testing.

---

## Table of Contents

1. [**Quick Reference**](#quick-reference) - Test priority overview
2. [**Test Plan 1: Basic Performance Metrics**](#test-plan-1-basic-performance-metrics) - Accuracy, precision, recall, F1
3. [**Test Plan 2: Per-Class Analysis**](#test-plan-2-per-class-analysis) - Identify weak classes
4. [**Test Plan 3: Confusion Matrix Analysis**](#test-plan-3-confusion-matrix-analysis) - Common misclassifications
5. [**Test Plan 4: Confidence Calibration**](#test-plan-4-confidence-calibration) - Is the model confident when correct?
6. [**Test Plan 5: Robustness Testing**](#test-plan-5-robustness-testing) - Image quality, lighting, angles
7. [**Test Plan 6: Real-World Scenario Testing**](#test-plan-6-real-world-scenario-testing) - Mobile photos, restaurant plates
8. [**Test Plan 7: Edge Case Testing**](#test-plan-7-edge-case-testing) - Ambiguous foods, similar classes
9. [**Test Plan 8: Cross-Dataset Validation**](#test-plan-8-cross-dataset-validation) - Generalization ability
10. [**Test Plan 9: Performance Benchmarking**](#test-plan-9-performance-benchmarking) - Speed and latency
11. [**Test Plan 10: A/B Comparison Testing**](#test-plan-10-ab-comparison-testing) - Compare different models

---

## Quick Reference

| Test Plan | Priority | Time | Purpose |
|-----------|----------|------|---------|
| **1. Basic Metrics** | 🔴 CRITICAL | 5 min | Overall accuracy, F1, precision, recall |
| **2. Per-Class Analysis** | 🔴 CRITICAL | 10 min | Find weak/strong classes |
| **3. Confusion Matrix** | 🟠 HIGH | 15 min | Identify common errors |
| **4. Confidence Calibration** | 🟠 HIGH | 10 min | Check if confidence matches accuracy |
| **5. Robustness Testing** | 🟡 MEDIUM | 30 min | Test on noisy/degraded images |
| **6. Real-World Testing** | 🟡 MEDIUM | 1 hour | Test on real photos |
| **7. Edge Cases** | 🟢 LOW | 30 min | Ambiguous/tricky samples |
| **8. Cross-Dataset** | 🟢 LOW | 20 min | Generalization check |
| **9. Performance Benchmark** | 🟡 MEDIUM | 15 min | Speed and memory usage |
| **10. A/B Comparison** | 🟡 MEDIUM | Variable | Compare models |

---

## Test Plan 1: Basic Performance Metrics

### Purpose
Measure fundamental model quality with standard ML metrics.

### Metrics to Calculate

1. **Overall Accuracy**: % of correct predictions
2. **Top-5 Accuracy**: % where correct class is in top 5 predictions
3. **Precision**: True positives / (True positives + False positives)
4. **Recall**: True positives / (True positives + False negatives)
5. **F1 Score**: Harmonic mean of precision and recall
6. **Macro-averaged metrics**: Average across all classes (treats each class equally)
7. **Weighted-averaged metrics**: Weighted by class support

### Script

```bash
python src/evaluation/test_basic_metrics.py \
    --model models/recognition/food-101_efficientnet_b3/best_accuracy.pt \
    --dataset food-101 \
    --split test \
    --device mps
```

### Expected Output

```
=================================================================
BASIC PERFORMANCE METRICS
=================================================================
Test Set Size: 25,250 images
Number of Classes: 101

Overall Metrics:
  Accuracy:         91.2%
  Top-5 Accuracy:   98.1%
  Precision:        90.8%
  Recall:           91.0%
  F1 Score:         90.9%

Macro-Averaged (treats all classes equally):
  Precision:        89.5%
  Recall:           90.1%
  F1 Score:         89.8%

Weighted-Averaged (by class support):
  Precision:        91.0%
  Recall:           91.2%
  F1 Score:         91.1%
=================================================================
```

### Success Criteria

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| Accuracy | >85% | >90% | >93% |
| Top-5 Accuracy | >95% | >97% | >99% |
| F1 Score | >83% | >88% | >91% |

---

## Test Plan 2: Per-Class Analysis

### Purpose
Identify which food classes the model performs well/poorly on.

### What to Analyze

1. **Per-class accuracy**: Accuracy for each of the 101 classes
2. **Best performing classes**: Top 10 classes with highest accuracy
3. **Worst performing classes**: Bottom 10 classes with lowest accuracy
4. **Class imbalance impact**: Correlation between class size and accuracy
5. **Precision vs Recall per class**: Classes with high FP vs high FN

### Script

```bash
python src/evaluation/test_per_class.py \
    --model models/recognition/food-101_efficientnet_b3/best_accuracy.pt \
    --dataset food-101 \
    --split test \
    --device mps \
    --output results/per_class_analysis.html
```

### Expected Output

```
=================================================================
PER-CLASS PERFORMANCE ANALYSIS
=================================================================

TOP 10 PERFORMING CLASSES:
  1. waffles              98.5% accuracy (197/200)
  2. french_fries         97.2% accuracy (194/200)
  3. ice_cream            96.8% accuracy (194/200)
  4. pizza                96.0% accuracy (192/200)
  5. donuts               95.5% accuracy (191/200)
  ...

BOTTOM 10 PERFORMING CLASSES:
  1. beef_carpaccio       72.5% accuracy (145/200)
  2. beef_tartare         74.0% accuracy (148/200)
  3. pork_chop            76.5% accuracy (153/200)
  4. filet_mignon         78.0% accuracy (156/200)
  5. steak                79.5% accuracy (159/200)
  ...

CLASSES WITH HIGH FALSE POSITIVES:
  - pizza: Often confused with other classes (38 FP)
  - chicken_curry: Confused with similar curries (32 FP)
  ...

CLASSES WITH HIGH FALSE NEGATIVES:
  - beef_carpaccio: Often missed (55 FN)
  - panna_cotta: Confused with desserts (47 FN)
  ...

Generated HTML report: results/per_class_analysis.html
=================================================================
```

### Action Items

- **Collect more data** for weak classes
- **Analyze misclassifications** for bottom performers
- **Consider class merging** for very similar classes (e.g., beef dishes)

---

## Test Plan 3: Confusion Matrix Analysis

### Purpose
Identify systematic misclassifications and confusable food pairs.

### What to Analyze

1. **Confusion matrix**: 101×101 heatmap showing predictions vs ground truth
2. **Most confused pairs**: Top 20 food pairs the model confuses
3. **One-sided confusion**: Classes A→B but not B→A
4. **Cluster analysis**: Groups of mutually confusable foods

### Script

```bash
python src/evaluation/test_confusion_matrix.py \
    --model models/recognition/food-101_efficientnet_b3/best_accuracy.pt \
    --dataset food-101 \
    --split test \
    --device mps \
    --output results/confusion_matrix.png \
    --top-k 20
```

### Expected Output

```
=================================================================
CONFUSION MATRIX ANALYSIS
=================================================================

TOP 20 CONFUSED PAIRS (Predicted → Actual):
  1. beef_carpaccio → beef_tartare      (42 times)
  2. beef_tartare → beef_carpaccio      (38 times)
  3. chicken_curry → chicken_quesadilla (28 times)
  4. chocolate_cake → chocolate_mousse  (25 times)
  5. spaghetti_bolognese → spaghetti_carbonara (23 times)
  ...

CONFUSION CLUSTERS:
  Cluster 1: Beef dishes
    - beef_carpaccio ↔ beef_tartare
    - filet_mignon ↔ steak ↔ prime_rib

  Cluster 2: Pasta dishes
    - spaghetti_bolognese ↔ spaghetti_carbonara
    - ravioli ↔ gnocchi

  Cluster 3: Chocolate desserts
    - chocolate_cake ↔ chocolate_mousse
    - chocolate_cake ↔ red_velvet_cake

Generated confusion matrix: results/confusion_matrix.png
=================================================================
```

### Action Items

- **Review confused pairs**: Are they actually similar? (validation error?)
- **Collect disambiguating features**: What distinguishes confused pairs?
- **Consider ensemble**: Different models for confusable groups

---

## Test Plan 4: Confidence Calibration

### Purpose
Verify that model confidence (softmax probability) correlates with actual accuracy.

### What to Measure

1. **Calibration curve**: Plot predicted confidence vs actual accuracy
2. **Expected Calibration Error (ECE)**: Average difference between confidence and accuracy
3. **Confidence distribution**: Histogram of prediction confidences
4. **Over/under-confident classes**: Which classes have miscalibrated confidence

### Script

```bash
python src/evaluation/test_confidence_calibration.py \
    --model models/recognition/food-101_efficientnet_b3/best_accuracy.pt \
    --dataset food-101 \
    --split test \
    --device mps \
    --output results/calibration_plot.png
```

### Expected Output

```
=================================================================
CONFIDENCE CALIBRATION ANALYSIS
=================================================================

Expected Calibration Error (ECE): 3.2%
  (Lower is better, <5% is well-calibrated)

Confidence Distribution:
  0.0-0.5:   12.3% of predictions
  0.5-0.7:   18.7% of predictions
  0.7-0.9:   35.2% of predictions
  0.9-1.0:   33.8% of predictions

Calibration by Confidence Bin:
  Confidence 0.9-1.0: 96.8% actual accuracy (under-confident)
  Confidence 0.8-0.9: 88.2% actual accuracy (well-calibrated)
  Confidence 0.7-0.8: 79.5% actual accuracy (well-calibrated)
  Confidence 0.5-0.7: 62.1% actual accuracy (over-confident)

OVER-CONFIDENT CLASSES (high confidence, low accuracy):
  1. beef_carpaccio: 82% avg confidence, 72% accuracy
  2. beef_tartare: 80% avg confidence, 74% accuracy
  ...

UNDER-CONFIDENT CLASSES (low confidence, high accuracy):
  1. waffles: 92% avg confidence, 98% accuracy
  2. pizza: 89% avg confidence, 96% accuracy
  ...

Generated calibration plot: results/calibration_plot.png
=================================================================
```

### Action Items

- **Apply temperature scaling** if ECE > 5%
- **Threshold adjustment**: Set confidence threshold for production
- **Reject low-confidence predictions** (e.g., < 50%)

---

## Test Plan 5: Robustness Testing

### Purpose
Test model performance under degraded conditions (blur, noise, lighting).

### Test Scenarios

1. **Gaussian blur**: Simulates out-of-focus images
2. **Gaussian noise**: Simulates low-light/high-ISO images
3. **JPEG compression**: Simulates heavily compressed images
4. **Brightness variations**: Dark/bright images
5. **Contrast variations**: Low/high contrast
6. **Rotation**: 90°, 180°, 270° rotations
7. **Occlusion**: Partially covered food items

### Script

```bash
python src/evaluation/test_robustness.py \
    --model models/recognition/food-101_efficientnet_b3/best_accuracy.pt \
    --dataset food-101 \
    --split test \
    --num-samples 1000 \
    --device mps \
    --output results/robustness_report.json
```

### Expected Output

```
=================================================================
ROBUSTNESS TESTING
=================================================================
Baseline Accuracy: 91.2%

Perturbation Results:
  Gaussian Blur (σ=2):          87.3% (-3.9%)
  Gaussian Blur (σ=5):          79.1% (-12.1%)
  Gaussian Noise (σ=0.1):       88.5% (-2.7%)
  Gaussian Noise (σ=0.2):       82.3% (-8.9%)
  JPEG Compression (Q=30):      89.2% (-2.0%)
  JPEG Compression (Q=10):      81.5% (-9.7%)
  Brightness +30%:              90.1% (-1.1%)
  Brightness -30%:              85.2% (-6.0%)
  Contrast +50%:                89.8% (-1.4%)
  Contrast -50%:                83.7% (-7.5%)
  Rotation 90°:                 88.4% (-2.8%)
  Occlusion 20%:                86.1% (-5.1%)
  Occlusion 40%:                76.8% (-14.4%)

Most Robust Classes:
  1. pizza:        95.2% avg across perturbations
  2. ice_cream:    94.1% avg across perturbations
  ...

Least Robust Classes:
  1. beef_carpaccio:  65.3% avg across perturbations
  2. panna_cotta:     68.7% avg across perturbations
  ...

Generated report: results/robustness_report.json
=================================================================
```

### Success Criteria

- **Blur (σ=2)**: < 5% accuracy drop
- **Noise (σ=0.1)**: < 5% accuracy drop
- **JPEG (Q=30)**: < 3% accuracy drop
- **Brightness ±30%**: < 8% accuracy drop

### Action Items

- **Add augmentation** for weak perturbations during training
- **Test-time augmentation (TTA)**: Average predictions across augmentations
- **Preprocessing**: Apply denoising/sharpening in production

---

## Test Plan 6: Real-World Scenario Testing

### Purpose
Test on actual real-world images (not from training distribution).

### Test Scenarios

1. **Mobile phone photos**: Personal food photos
2. **Restaurant photos**: Professional food photography
3. **Home cooking**: Home-made food images
4. **Multiple items**: Plates with multiple food items
5. **Partial views**: Foods partially visible
6. **Different angles**: Top-down, side, 45° angle
7. **Various backgrounds**: Table, plate, hand-held

### Script

```bash
# Collect real-world test set first
mkdir -p data/real_world_test
# ... add your own food photos ...

python src/evaluation/test_real_world.py \
    --model models/recognition/food-101_efficientnet_b3/best_accuracy.pt \
    --test-dir data/real_world_test \
    --device mps \
    --output results/real_world_report.html
```

### Expected Output

```
=================================================================
REAL-WORLD SCENARIO TESTING
=================================================================
Test Set: 500 real-world images

Overall Performance:
  Accuracy: 78.2% (lower than test set, expected)
  Top-5 Accuracy: 92.1%
  Confidence: 76.3% (avg)

By Scenario:
  Mobile photos:      81.3% accuracy (120 images)
  Restaurant photos:  89.7% accuracy (100 images)
  Home cooking:       72.5% accuracy (150 images)
  Multiple items:     68.1% accuracy (80 images)
  Partial views:      65.3% accuracy (50 images)

By Angle:
  Top-down:           85.2% accuracy
  45° angle:          77.8% accuracy
  Side view:          69.4% accuracy

Common Failures:
  - Multiple items in same image (confusion)
  - Home-made versions differ from restaurant
  - Unusual presentations/plating
  - Poor lighting in mobile photos

Generated report: results/real_world_report.html
=================================================================
```

### Success Criteria

- **Mobile photos**: > 75% accuracy
- **Restaurant photos**: > 85% accuracy
- **Top-5 accuracy**: > 90% (user can correct)

### Action Items

- **Fine-tune on real-world data**: Collect and annotate failure cases
- **Add data augmentation**: Simulate real-world conditions
- **Multi-item detection**: Use SAM2 segmentation first

---

## Test Plan 7: Edge Case Testing

### Purpose
Test model behavior on ambiguous, rare, or tricky cases.

### Test Scenarios

1. **Similar looking foods**: Foods that look nearly identical
2. **Regional variations**: Same dish, different regions
3. **Presentation variations**: Same food, different plating
4. **Mixed dishes**: Combinations not in training set
5. **Partial/incomplete dishes**: Half-eaten food
6. **Non-food images**: Should reject with low confidence

### Script

```bash
python src/evaluation/test_edge_cases.py \
    --model models/recognition/food-101_efficientnet_b3/best_accuracy.pt \
    --dataset food-101 \
    --device mps \
    --output results/edge_cases_report.html
```

### Expected Output

```
=================================================================
EDGE CASE TESTING
=================================================================

SIMILAR LOOKING FOODS (Confusion Test):
  Test: beef_carpaccio vs beef_tartare (50 samples each)
  Result: 68% correctly distinguished (needs improvement)

  Test: spaghetti_bolognese vs spaghetti_carbonara (50 each)
  Result: 82% correctly distinguished (acceptable)

REGIONAL VARIATIONS:
  Test: Italian pizza vs American pizza vs Chicago deep-dish
  Result: All classified as "pizza" (correct), 95% confidence

PRESENTATION VARIATIONS:
  Test: Same food, 5 different plating styles
  Result: 88% consistent classification across variations

NON-FOOD REJECTION:
  Test: 100 random non-food images
  Result: 87% correctly rejected (confidence < 0.5)
  Issue: 13% false positives (classified as food)

CONFIDENCE DISTRIBUTION ON EDGE CASES:
  Similar foods:       67.2% avg confidence
  Normal foods:        85.1% avg confidence
  Non-food:            32.5% avg confidence

Generated report: results/edge_cases_report.html
=================================================================
```

### Action Items

- **Collect hard negatives**: Add confusable pairs to training
- **Add "unknown" class**: Train to reject non-food items
- **Threshold tuning**: Reject predictions below confidence threshold

---

## Test Plan 8: Cross-Dataset Validation

### Purpose
Test if model generalizes beyond training dataset.

### Test Scenarios

1. **Food-101 → Nutrition5k**: Train on Food-101, test on Nutrition5k
2. **Nutrition5k → Food-101**: Train on Nutrition5k, test on Food-101
3. **Combined → Held-out dataset**: Test on different dataset entirely
4. **Domain adaptation**: Fine-tune on small amount of target data

### Script

```bash
# Test Food-101 model on Nutrition5k data
python src/evaluation/test_cross_dataset.py \
    --model models/recognition/food-101_efficientnet_b3/best_accuracy.pt \
    --source-dataset food-101 \
    --target-dataset nutrition5k \
    --device mps \
    --output results/cross_dataset_report.json
```

### Expected Output

```
=================================================================
CROSS-DATASET VALIDATION
=================================================================
Source Dataset: food-101 (trained)
Target Dataset: nutrition5k (held-out)

Performance on Target Dataset:
  Accuracy: 45.2% (expected drop, different label spaces)
  Top-5 Accuracy: 67.8%

Transferable Classes (present in both):
  pizza:           93.2% accuracy (good transfer)
  salad:           87.5% accuracy (good transfer)
  dessert:         72.1% accuracy (moderate transfer)
  ...

Non-transferable Classes:
  nutrition5k-specific classes have low recall (expected)

Domain Adaptation (with 100 target samples):
  Accuracy after fine-tuning: 68.3% (+23.1%)

Recommendation:
  - Use "combined" dataset for better generalization
  - Fine-tune on target domain if deploying
  - Consider domain-invariant features

Generated report: results/cross_dataset_report.json
=================================================================
```

### Success Criteria

- **Related datasets**: > 60% accuracy on target
- **After fine-tuning (100 samples)**: > 75% accuracy

---

## Test Plan 9: Performance Benchmarking

### Purpose
Measure inference speed, latency, memory usage, and throughput.

### What to Measure

1. **Inference latency**: Time per image (ms)
2. **Throughput**: Images per second
3. **Memory usage**: RAM and GPU memory
4. **Model size**: File size (MB)
5. **Batch size impact**: Latency vs throughput trade-off
6. **Device comparison**: MPS vs CPU performance

### Script

```bash
python src/evaluation/test_performance.py \
    --model models/recognition/food-101_efficientnet_b3/best_accuracy.pt \
    --device mps \
    --batch-sizes 1,4,8,16,32 \
    --num-iterations 100 \
    --output results/performance_report.json
```

### Expected Output

```
=================================================================
PERFORMANCE BENCHMARKING
=================================================================
Model: food-101_efficientnet_b3
Device: MPS (Apple Silicon)
Model Size: 47.2 MB
Model Parameters: 12.3M

LATENCY (per image):
  Batch Size 1:    15.2 ms
  Batch Size 4:     8.7 ms (4.2x speedup per image)
  Batch Size 8:     7.1 ms (2.1x speedup per image)
  Batch Size 16:    6.8 ms (1.05x speedup per image)
  Batch Size 32:    7.2 ms (0.94x speedup, worse!)

THROUGHPUT:
  Batch Size 1:    65.8 images/sec
  Batch Size 4:   114.9 images/sec
  Batch Size 8:   140.8 images/sec
  Batch Size 16:  147.1 images/sec (optimal)
  Batch Size 32:  138.9 images/sec (memory bottleneck)

MEMORY USAGE:
  Model Memory: 188 MB
  Peak Memory (batch=1):   512 MB
  Peak Memory (batch=16): 1.8 GB
  Peak Memory (batch=32): 3.2 GB

DEVICE COMPARISON:
  MPS:  15.2 ms/image
  CPU:  42.7 ms/image (2.8x slower)

BREAKDOWN (batch=1):
  Data loading:      2.1 ms (13.8%)
  Preprocessing:     1.8 ms (11.8%)
  Inference:        10.5 ms (69.1%)
  Post-processing:   0.8 ms  (5.3%)

Generated report: results/performance_report.json
=================================================================
```

### Success Criteria

| Metric | Requirement | Target |
|--------|-------------|--------|
| Latency (single) | < 50ms | < 20ms |
| Throughput (batch=16) | > 100 img/s | > 150 img/s |
| Memory | < 4GB | < 2GB |
| Model size | < 100MB | < 50MB |

### Action Items

- **Optimize batch size**: Use optimal batch size (16 in example)
- **Model quantization**: Reduce model size with INT8
- **ONNX export**: Convert to ONNX for production
- **TensorRT**: Use TensorRT for NVIDIA GPUs

---

## Test Plan 10: A/B Comparison Testing

### Purpose
Compare multiple models or training configurations.

### Comparison Dimensions

1. **Model architecture**: EfficientNet-B3 vs ResNet-50
2. **Training method**: Single-stage vs two-stage
3. **Augmentation**: With vs without mixup/cutmix
4. **Dataset**: Food-101 vs Combined
5. **Ensemble**: Single model vs 3-model ensemble

### Script

```bash
python src/evaluation/test_ab_comparison.py \
    --model-a models/recognition/food-101_efficientnet_b3/best_accuracy.pt \
    --model-b models/recognition/food-101_resnet50/best_accuracy.pt \
    --dataset food-101 \
    --split test \
    --device mps \
    --output results/ab_comparison.html
```

### Expected Output

```
=================================================================
A/B MODEL COMPARISON
=================================================================

Model A: food-101_efficientnet_b3
Model B: food-101_resnet50

OVERALL PERFORMANCE:
                    Model A    Model B    Δ
  Accuracy          91.2%      88.5%      +2.7%
  Top-5 Accuracy    98.1%      96.8%      +1.3%
  F1 Score          90.9%      87.8%      +3.1%
  Avg Confidence    85.1%      82.3%      +2.8%

INFERENCE SPEED:
  Latency           15.2 ms    18.7 ms    -3.5 ms (A faster)
  Throughput        147 img/s  128 img/s  +19 img/s (A faster)
  Model Size        47.2 MB    97.5 MB    -50.3 MB (A smaller)

PER-CLASS COMPARISON:
  Classes where Model A wins: 67/101
  Classes where Model B wins: 28/101
  Tied classes: 6/101

CLASSES WHERE MODEL B SIGNIFICANTLY BETTER:
  1. steak:        A: 79.5%  B: 88.2%  (+8.7%)
  2. filet_mignon: A: 78.0%  B: 85.1%  (+7.1%)
  ...

CLASSES WHERE MODEL A SIGNIFICANTLY BETTER:
  1. pizza:        A: 96.0%  B: 89.2%  (+6.8%)
  2. waffles:      A: 98.5%  B: 92.1%  (+6.4%)
  ...

RECOMMENDATION:
  → Use Model A (EfficientNet-B3)
  Reasons:
    - Higher overall accuracy (+2.7%)
    - Faster inference (15.2ms vs 18.7ms)
    - Smaller model size (47.2MB vs 97.5MB)
    - Better on 67% of classes

  Consider Model B for:
    - Better performance on meat dishes

Generated report: results/ab_comparison.html
=================================================================
```

---

## Testing Workflow

### Phase 1: Essential Tests (30 minutes)

Run these tests immediately after training:

```bash
# 1. Basic metrics (5 min)
python src/evaluation/test_basic_metrics.py --model <path> --dataset food-101

# 2. Per-class analysis (10 min)
python src/evaluation/test_per_class.py --model <path> --dataset food-101

# 3. Confusion matrix (15 min)
python src/evaluation/test_confusion_matrix.py --model <path> --dataset food-101
```

**Decision Point**: If accuracy > 90% and no major issues → proceed to Phase 2

---

### Phase 2: Quality Assurance (1-2 hours)

```bash
# 4. Confidence calibration
python src/evaluation/test_confidence_calibration.py --model <path>

# 5. Robustness testing
python src/evaluation/test_robustness.py --model <path> --num-samples 1000

# 6. Performance benchmark
python src/evaluation/test_performance.py --model <path>
```

**Decision Point**: If model is robust and fast → proceed to Phase 3

---

### Phase 3: Real-World Validation (variable time)

```bash
# 7. Real-world testing (requires collecting test images)
python src/evaluation/test_real_world.py --model <path> --test-dir data/real_world

# 8. Edge case testing
python src/evaluation/test_edge_cases.py --model <path>
```

**Decision Point**: If real-world performance acceptable → deploy!

---

## Automated Test Suite

Run all critical tests with one command:

```bash
# Run all essential tests
bash scripts/test_model_quality.sh \
    models/recognition/food-101_efficientnet_b3/best_accuracy.pt

# Generates:
# - results/test_report.html (comprehensive HTML report)
# - results/test_summary.json (machine-readable results)
# - results/test_plots/ (all visualization plots)
```

---

## Continuous Testing

For production models, set up continuous testing:

```bash
# Weekly automated testing
cron: 0 0 * * 0  # Every Sunday at midnight
  bash scripts/weekly_model_test.sh
  # Sends alert if performance drops > 5%
```

---

## Summary: Which Tests to Run?

### For Development
- ✅ Test Plan 1: Basic Metrics
- ✅ Test Plan 2: Per-Class Analysis
- ✅ Test Plan 3: Confusion Matrix

### For Quality Assurance
- ✅ Test Plan 4: Confidence Calibration
- ✅ Test Plan 5: Robustness Testing
- ✅ Test Plan 9: Performance Benchmark

### Before Production Deployment
- ✅ Test Plan 6: Real-World Testing
- ✅ Test Plan 7: Edge Case Testing

### For Model Comparison
- ✅ Test Plan 10: A/B Comparison

### For Research/Generalization
- ✅ Test Plan 8: Cross-Dataset Validation

---

**Created**: 2025-11-04
**Status**: Ready for implementation
**Next Steps**: Create evaluation scripts for each test plan
**Related**: `HIGH_QUALITY_TRAINING_STRATEGY.md`, `DATASETS_AND_INCREMENTAL_TRAINING.md`
