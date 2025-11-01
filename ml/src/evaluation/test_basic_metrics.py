#!/usr/bin/env python3
"""
Test Plan 1: Basic Performance Metrics

Measures fundamental model quality:
- Overall accuracy
- Top-5 accuracy
- Precision, Recall, F1
- Macro and weighted averages

Usage:
    python src/evaluation/test_basic_metrics.py \
        --model models/recognition/food-101_efficientnet_b3/best_accuracy.pt \
        --dataset food-101 \
        --device mps
"""
import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config
from data_process.classification_dataset import FoodClassificationDataset, collate_fn
from data_process.food_labels import FoodLabelManager
from models import FoodRecognitionModel
from training.classification_metrics import ClassificationMetrics


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test Plan 1: Basic Performance Metrics")

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model (.pt file)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="food-101",
        choices=["food-101", "nutrition5k", "combined"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=config.processed_data_path,
        help="Path to processed data",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.device,
        choices=["cuda", "mps", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for results (JSON)",
    )

    return parser.parse_args()


def compute_top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """Calculate top-k accuracy"""
    with torch.no_grad():
        # Get top k predictions
        _, top_k_preds = torch.topk(logits, k, dim=1)
        # Check if target is in top k
        correct = top_k_preds.eq(targets.view(-1, 1).expand_as(top_k_preds))
        # Calculate accuracy
        top_k_acc = correct.sum().item() / targets.size(0)
    return top_k_acc


def evaluate_model(model, dataloader, device, num_classes):
    """Evaluate model and compute metrics"""
    model.eval()

    # Initialize metrics
    metrics_macro = ClassificationMetrics(num_classes, average="macro")
    metrics_weighted = ClassificationMetrics(num_classes, average="weighted")

    all_logits = []
    all_targets = []
    total_samples = 0

    logger.info("Running evaluation...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            logits = model(images)

            # Update metrics
            metrics_macro.update(logits, labels)
            metrics_weighted.update(logits, labels)

            # Store for top-k accuracy
            all_logits.append(logits.cpu())
            all_targets.append(labels.cpu())

            total_samples += labels.size(0)

    # Compute metrics
    results_macro = metrics_macro.compute()
    results_weighted = metrics_weighted.compute()

    # Compute top-5 accuracy
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    top5_acc = compute_top_k_accuracy(all_logits, all_targets, k=5)

    # Compile results
    results = {
        "total_samples": total_samples,
        "num_classes": num_classes,
        "overall": {
            "accuracy": results_macro["accuracy"],
            "top5_accuracy": top5_acc,
        },
        "macro_averaged": {
            "precision": results_macro["precision"],
            "recall": results_macro["recall"],
            "f1": results_macro["f1"],
        },
        "weighted_averaged": {
            "precision": results_weighted["precision"],
            "recall": results_weighted["recall"],
            "f1": results_weighted["f1"],
        },
    }

    return results


def print_results(results):
    """Print results in formatted output"""
    logger.info("=" * 65)
    logger.info("BASIC PERFORMANCE METRICS")
    logger.info("=" * 65)
    logger.info(f"Test Set Size: {results['total_samples']:,} images")
    logger.info(f"Number of Classes: {results['num_classes']}")
    logger.info("")

    logger.info("Overall Metrics:")
    logger.info(f"  Accuracy:         {results['overall']['accuracy']:.1%}")
    logger.info(f"  Top-5 Accuracy:   {results['overall']['top5_accuracy']:.1%}")
    logger.info("")

    logger.info("Macro-Averaged (treats all classes equally):")
    logger.info(f"  Precision:        {results['macro_averaged']['precision']:.1%}")
    logger.info(f"  Recall:           {results['macro_averaged']['recall']:.1%}")
    logger.info(f"  F1 Score:         {results['macro_averaged']['f1']:.1%}")
    logger.info("")

    logger.info("Weighted-Averaged (by class support):")
    logger.info(f"  Precision:        {results['weighted_averaged']['precision']:.1%}")
    logger.info(f"  Recall:           {results['weighted_averaged']['recall']:.1%}")
    logger.info(f"  F1 Score:         {results['weighted_averaged']['f1']:.1%}")
    logger.info("=" * 65)


def assess_quality(results):
    """Provide quality assessment"""
    accuracy = results["overall"]["accuracy"]
    top5_acc = results["overall"]["top5_accuracy"]
    f1 = results["macro_averaged"]["f1"]

    logger.info("")
    logger.info("Quality Assessment:")

    # Accuracy assessment
    if accuracy >= 0.93:
        logger.info("  ✅ Accuracy: EXCELLENT (≥93%)")
    elif accuracy >= 0.90:
        logger.info("  ✅ Accuracy: GOOD (≥90%)")
    elif accuracy >= 0.85:
        logger.info("  ⚠️  Accuracy: ACCEPTABLE (≥85%)")
    else:
        logger.info("  ❌ Accuracy: NEEDS IMPROVEMENT (<85%)")

    # Top-5 accuracy assessment
    if top5_acc >= 0.99:
        logger.info("  ✅ Top-5 Accuracy: EXCELLENT (≥99%)")
    elif top5_acc >= 0.97:
        logger.info("  ✅ Top-5 Accuracy: GOOD (≥97%)")
    elif top5_acc >= 0.95:
        logger.info("  ⚠️  Top-5 Accuracy: ACCEPTABLE (≥95%)")
    else:
        logger.info("  ❌ Top-5 Accuracy: NEEDS IMPROVEMENT (<95%)")

    # F1 assessment
    if f1 >= 0.91:
        logger.info("  ✅ F1 Score: EXCELLENT (≥91%)")
    elif f1 >= 0.88:
        logger.info("  ✅ F1 Score: GOOD (≥88%)")
    elif f1 >= 0.83:
        logger.info("  ⚠️  F1 Score: ACCEPTABLE (≥83%)")
    else:
        logger.info("  ❌ F1 Score: NEEDS IMPROVEMENT (<83%)")

    logger.info("")


def main():
    """Main evaluation function"""
    args = parse_args()

    # Load label manager
    label_mapping_path = args.model.parent / "label_mapping.json"
    if not label_mapping_path.exists():
        logger.error(f"Label mapping not found: {label_mapping_path}")
        logger.error("Please ensure label_mapping.json exists in model directory")
        return

    logger.info(f"Loading label mapping from {label_mapping_path}")
    label_manager = FoodLabelManager.load_mapping(label_mapping_path)

    # Load model
    logger.info(f"Loading model from {args.model}")
    model = FoodRecognitionModel.load(args.model, device=args.device)
    model.eval()

    # Create dataset
    data_file = args.data_dir / f"{args.split}.parquet"
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        return

    logger.info(f"Loading {args.split} dataset from {data_file}")
    dataset = FoodClassificationDataset(
        data_file=data_file,
        label_manager=label_manager,
        mode="val",  # Use val mode (no augmentation)
        include_nutrition=False,
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Single worker for evaluation
        collate_fn=collate_fn,
    )

    logger.info(f"Dataset: {len(dataset)} samples")
    logger.info(f"Number of classes: {label_manager.num_classes}")
    logger.info("")

    # Evaluate
    results = evaluate_model(model, dataloader, args.device, label_manager.num_classes)

    # Print results
    print_results(results)

    # Assess quality
    assess_quality(results)

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
