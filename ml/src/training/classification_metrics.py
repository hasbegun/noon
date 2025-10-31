"""
Metrics for food classification tasks
"""
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from loguru import logger


class ClassificationMetrics:
    """Calculate classification metrics: accuracy, precision, recall, F1"""

    def __init__(self, num_classes: int, average: str = "macro"):
        """
        Initialize metrics calculator

        Args:
            num_classes: Number of classes
            average: Averaging method ('macro', 'weighted', 'micro')
        """
        self.num_classes = num_classes
        self.average = average
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.correct = 0
        self.total = 0
        self.per_class_correct = torch.zeros(self.num_classes)
        self.per_class_total = torch.zeros(self.num_classes)
        self.per_class_predicted = torch.zeros(self.num_classes)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with batch predictions

        Args:
            predictions: Model predictions (B,) or logits (B, num_classes)
            targets: Ground truth labels (B,)
        """
        # Move to CPU first to save GPU memory
        predictions = predictions.detach().cpu()
        targets = targets.detach().cpu()

        # Convert logits to predictions if needed
        if predictions.dim() == 2:
            predictions = torch.argmax(predictions, dim=1)

        # Overall accuracy
        self.correct += (predictions == targets).sum().item()
        self.total += targets.size(0)

        # Per-class metrics
        for c in range(self.num_classes):
            # True positives
            self.per_class_correct[c] += ((predictions == c) & (targets == c)).sum().item()
            # Total ground truth for this class
            self.per_class_total[c] += (targets == c).sum().item()
            # Total predicted for this class
            self.per_class_predicted[c] += (predictions == c).sum().item()

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics

        Returns:
            Dictionary with metrics
        """
        # Overall accuracy
        accuracy = self.correct / max(self.total, 1)

        # Per-class precision, recall, F1
        precision = torch.zeros(self.num_classes)
        recall = torch.zeros(self.num_classes)
        f1 = torch.zeros(self.num_classes)

        for c in range(self.num_classes):
            # Precision: TP / (TP + FP)
            if self.per_class_predicted[c] > 0:
                precision[c] = self.per_class_correct[c] / self.per_class_predicted[c]
            else:
                precision[c] = 0.0

            # Recall: TP / (TP + FN)
            if self.per_class_total[c] > 0:
                recall[c] = self.per_class_correct[c] / self.per_class_total[c]
            else:
                recall[c] = 0.0

            # F1: 2 * (precision * recall) / (precision + recall)
            if precision[c] + recall[c] > 0:
                f1[c] = 2 * (precision[c] * recall[c]) / (precision[c] + recall[c])
            else:
                f1[c] = 0.0

        # Average metrics
        if self.average == "macro":
            # Macro average: simple mean across classes
            avg_precision = precision.mean().item()
            avg_recall = recall.mean().item()
            avg_f1 = f1.mean().item()

        elif self.average == "weighted":
            # Weighted average: weighted by class support
            weights = self.per_class_total / max(self.per_class_total.sum(), 1)
            avg_precision = (precision * weights).sum().item()
            avg_recall = (recall * weights).sum().item()
            avg_f1 = (f1 * weights).sum().item()

        elif self.average == "micro":
            # Micro average: aggregate then calculate
            total_tp = self.per_class_correct.sum()
            total_pred = self.per_class_predicted.sum()
            total_true = self.per_class_total.sum()

            avg_precision = (total_tp / max(total_pred, 1)).item()
            avg_recall = (total_tp / max(total_true, 1)).item()

            if avg_precision + avg_recall > 0:
                avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
            else:
                avg_f1 = 0.0
        else:
            raise ValueError(f"Unknown average method: {self.average}")

        return {
            "accuracy": accuracy,
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
            "per_class_precision": precision.numpy(),
            "per_class_recall": recall.numpy(),
            "per_class_f1": f1.numpy(),
        }

    def get_top_k_accuracy(self, logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
        """
        Calculate top-k accuracy

        Args:
            logits: Model logits (B, num_classes)
            targets: Ground truth labels (B,)
            k: Top k predictions to consider

        Returns:
            Top-k accuracy
        """
        with torch.no_grad():
            # Get top k predictions
            _, top_k_pred = logits.topk(k, dim=1, largest=True, sorted=True)

            # Check if target is in top k
            targets = targets.view(-1, 1).expand_as(top_k_pred)
            correct = top_k_pred.eq(targets).any(dim=1).sum().item()

            accuracy = correct / logits.size(0)

        return accuracy


def print_classification_metrics(
    metrics: Dict[str, float],
    prefix: str = "",
    top_classes: Optional[list] = None,
    num_top_classes: int = 5
):
    """
    Print classification metrics in a readable format

    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for log messages (e.g., 'Train', 'Val')
        top_classes: List of class names (optional)
        num_top_classes: Number of top/bottom classes to show
    """
    logger.info(
        f"{prefix} Metrics - "
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | "
        f"F1: {metrics['f1']:.4f}"
    )

    # Show per-class metrics if class names provided
    if top_classes is not None and "per_class_f1" in metrics:
        per_class_f1 = metrics["per_class_f1"]

        # Get top performing classes
        top_indices = per_class_f1.argsort()[-num_top_classes:][::-1]
        logger.info(f"{prefix} Top {num_top_classes} classes by F1:")
        for idx in top_indices:
            if idx < len(top_classes):
                logger.info(f"  {top_classes[idx]}: {per_class_f1[idx]:.4f}")

        # Get worst performing classes
        worst_indices = per_class_f1.argsort()[:num_top_classes]
        logger.info(f"{prefix} Bottom {num_top_classes} classes by F1:")
        for idx in worst_indices:
            if idx < len(top_classes):
                logger.info(f"  {top_classes[idx]}: {per_class_f1[idx]:.4f}")


class NutritionRegressionMetrics:
    """Metrics for nutrition value prediction (regression task)"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.predictions = {
            'calories': [],
            'protein_g': [],
            'carb_g': [],
            'fat_g': [],
            'mass_g': [],
        }
        self.targets = {
            'calories': [],
            'protein_g': [],
            'carb_g': [],
            'fat_g': [],
            'mass_g': [],
        }

    def update(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        """
        Update metrics with batch predictions

        Args:
            predictions: Dictionary of predicted nutrition values
            targets: Dictionary of target nutrition values
        """
        for key in self.predictions.keys():
            if key in predictions and key in targets:
                self.predictions[key].append(predictions[key].cpu())
                self.targets[key].append(targets[key].cpu())

    def compute(self) -> Dict[str, float]:
        """Compute regression metrics (MAE, RMSE, R2)"""
        metrics = {}

        for key in self.predictions.keys():
            if len(self.predictions[key]) == 0:
                continue

            pred = torch.cat(self.predictions[key])
            target = torch.cat(self.targets[key])

            # Mean Absolute Error
            mae = (pred - target).abs().mean().item()

            # Root Mean Squared Error
            rmse = ((pred - target) ** 2).mean().sqrt().item()

            # R-squared
            ss_res = ((target - pred) ** 2).sum()
            ss_tot = ((target - target.mean()) ** 2).sum()
            r2 = (1 - ss_res / max(ss_tot, 1e-6)).item()

            metrics[f"{key}_mae"] = mae
            metrics[f"{key}_rmse"] = rmse
            metrics[f"{key}_r2"] = r2

        # Overall average MAE
        if len(metrics) > 0:
            mae_values = [v for k, v in metrics.items() if k.endswith('_mae')]
            metrics['avg_mae'] = sum(mae_values) / len(mae_values)

        return metrics


def print_nutrition_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Print nutrition regression metrics"""
    logger.info(f"{prefix} Nutrition Metrics:")

    for nutrient in ['calories', 'protein_g', 'carb_g', 'fat_g', 'mass_g']:
        mae_key = f"{nutrient}_mae"
        rmse_key = f"{nutrient}_rmse"
        r2_key = f"{nutrient}_r2"

        if mae_key in metrics:
            logger.info(
                f"  {nutrient}: "
                f"MAE={metrics[mae_key]:.2f}, "
                f"RMSE={metrics[rmse_key]:.2f}, "
                f"R²={metrics[r2_key]:.4f}"
            )

    if 'avg_mae' in metrics:
        logger.info(f"  Average MAE: {metrics['avg_mae']:.2f}")
