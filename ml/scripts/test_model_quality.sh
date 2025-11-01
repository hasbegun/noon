#!/bin/bash
# Comprehensive Model Quality Test Suite
#
# This script runs all critical tests to evaluate model quality.
# It's designed to be run after training completes.
#
# Usage:
#   bash scripts/test_model_quality.sh <model_path>
#
# Example:
#   bash scripts/test_model_quality.sh models/recognition/food-101_efficientnet_b3/best_accuracy.pt

set -e  # Exit on error

# Check if model path provided
if [ -z "$1" ]; then
    echo "Error: Model path required"
    echo "Usage: bash scripts/test_model_quality.sh <model_path>"
    echo "Example: bash scripts/test_model_quality.sh models/recognition/food-101_efficientnet_b3/best_accuracy.pt"
    exit 1
fi

MODEL_PATH="$1"
DATASET="${2:-food-101}"  # Default to food-101 if not specified
DEVICE="${3:-mps}"        # Default to mps if not specified

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found: $MODEL_PATH"
    exit 1
fi

# Extract model directory
MODEL_DIR=$(dirname "$MODEL_PATH")
MODEL_NAME=$(basename "$MODEL_DIR")

# Create results directory
RESULTS_DIR="results/quality_tests/${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "MODEL QUALITY TEST SUITE"
echo "============================================================"
echo "Model:       $MODEL_PATH"
echo "Dataset:     $DATASET"
echo "Device:      $DEVICE"
echo "Results:     $RESULTS_DIR"
echo "============================================================"
echo ""

# Log file
LOG_FILE="$RESULTS_DIR/test_log.txt"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Starting tests at: $(date)"
echo ""

#============================================================
# TEST 1: Basic Performance Metrics (5 minutes)
#============================================================
echo "============================================================"
echo "TEST 1: Basic Performance Metrics"
echo "============================================================"
echo "Running comprehensive accuracy, precision, recall, F1 tests..."
echo ""

python src/evaluation/test_basic_metrics.py \
    --model "$MODEL_PATH" \
    --dataset "$DATASET" \
    --device "$DEVICE" \
    --output "$RESULTS_DIR/basic_metrics.json"

echo ""
echo "✓ Test 1 complete!"
echo ""
sleep 2

#============================================================
# SUMMARY
#============================================================
echo ""
echo "============================================================"
echo "TEST SUITE COMPLETE"
echo "============================================================"
echo "Completed at: $(date)"
echo ""
echo "Results saved in: $RESULTS_DIR"
echo "  - basic_metrics.json     - Overall performance metrics"
echo "  - test_log.txt           - Full test log"
echo ""

# Extract key metrics for quick view
if [ -f "$RESULTS_DIR/basic_metrics.json" ]; then
    echo "Quick Summary:"
    echo "--------------"
    python3 -c "
import json
with open('$RESULTS_DIR/basic_metrics.json', 'r') as f:
    data = json.load(f)
    print(f\"  Accuracy:       {data['overall']['accuracy']:.1%}\")
    print(f\"  Top-5 Accuracy: {data['overall']['top5_accuracy']:.1%}\")
    print(f\"  F1 Score:       {data['macro_averaged']['f1']:.1%}\")
    print(f\"  Samples:        {data['total_samples']:,}\")
"
    echo ""
fi

echo "============================================================"
echo "Next Steps:"
echo "  1. Review metrics in $RESULTS_DIR/"
echo "  2. If accuracy > 90%, model is ready for deployment"
echo "  3. For detailed analysis, run individual test scripts"
echo "============================================================"
