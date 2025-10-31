"""
Metrics for evaluating segmentation model performance
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple
import numpy as np
from loguru import logger


class SegmentationMetrics:
    """
    Calculate segmentation metrics: F1, IoU, Precision, Recall
    """

    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Threshold for converting predictions to binary masks
        """
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_tn = 0
        self.num_samples = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with new batch

        Args:
            predictions: Model predictions (logits or probabilities) (B, 1, H, W)
            targets: Ground truth masks (B, 1, H, W) or (B, H, W)
        """
        # Convert predictions to probabilities
        if predictions.min() < 0 or predictions.max() > 1:
            predictions = torch.sigmoid(predictions)

        # Binarize predictions
        pred_binary = (predictions > self.threshold).float()

        # Ensure targets have same shape
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        # Flatten for metric calculation
        pred_flat = pred_binary.view(-1)
        target_flat = targets.view(-1)

        # Calculate confusion matrix components
        tp = ((pred_flat == 1) & (target_flat == 1)).sum().item()
        fp = ((pred_flat == 1) & (target_flat == 0)).sum().item()
        fn = ((pred_flat == 0) & (target_flat == 1)).sum().item()
        tn = ((pred_flat == 0) & (target_flat == 0)).sum().item()

        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn
        self.total_tn += tn
        self.num_samples += predictions.shape[0]

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics

        Returns:
            Dictionary with precision, recall, f1, iou, accuracy
        """
        # Avoid division by zero
        eps = 1e-7

        # Precision: TP / (TP + FP)
        precision = self.total_tp / (self.total_tp + self.total_fp + eps)

        # Recall: TP / (TP + FN)
        recall = self.total_tp / (self.total_tp + self.total_fn + eps)

        # F1: 2 * (Precision * Recall) / (Precision + Recall)
        f1 = 2 * (precision * recall) / (precision + recall + eps)

        # IoU: TP / (TP + FP + FN)
        iou = self.total_tp / (self.total_tp + self.total_fp + self.total_fn + eps)

        # Accuracy: (TP + TN) / (TP + TN + FP + FN)
        total = self.total_tp + self.total_tn + self.total_fp + self.total_fn
        accuracy = (self.total_tp + self.total_tn) / (total + eps)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou,
            'accuracy': accuracy,
        }

    def get_confusion_matrix(self) -> Dict[str, int]:
        """Get confusion matrix components"""
        return {
            'tp': self.total_tp,
            'fp': self.total_fp,
            'fn': self.total_fn,
            'tn': self.total_tn,
        }


def calculate_iou_score(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calculate IoU (Intersection over Union) score

    Args:
        predictions: Model predictions (B, 1, H, W)
        targets: Ground truth masks (B, 1, H, W) or (B, H, W)
        threshold: Threshold for binarization

    Returns:
        IoU score
    """
    # Convert to probabilities
    if predictions.min() < 0 or predictions.max() > 1:
        predictions = torch.sigmoid(predictions)

    pred_binary = (predictions > threshold).float()

    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    targets = targets.float()

    # Calculate intersection and union
    intersection = (pred_binary * targets).sum()
    union = pred_binary.sum() + targets.sum() - intersection

    iou = intersection / (union + 1e-7)
    return iou.item()


def calculate_dice_score(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calculate Dice coefficient

    Args:
        predictions: Model predictions (B, 1, H, W)
        targets: Ground truth masks (B, 1, H, W) or (B, H, W)
        threshold: Threshold for binarization

    Returns:
        Dice score
    """
    # Convert to probabilities
    if predictions.min() < 0 or predictions.max() > 1:
        predictions = torch.sigmoid(predictions)

    pred_binary = (predictions > threshold).float()

    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    targets = targets.float()

    # Calculate Dice
    intersection = (pred_binary * targets).sum()
    dice = (2. * intersection) / (pred_binary.sum() + targets.sum() + 1e-7)

    return dice.item()


def calculate_f1_score(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calculate F1 score (same as Dice for binary segmentation)

    Args:
        predictions: Model predictions (B, 1, H, W)
        targets: Ground truth masks (B, 1, H, W) or (B, H, W)
        threshold: Threshold for binarization

    Returns:
        F1 score
    """
    return calculate_dice_score(predictions, targets, threshold)


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Pretty print metrics

    Args:
        metrics: Dictionary of metric name -> value
        prefix: Prefix for log message (e.g., "Train" or "Val")
    """
    metric_str = " | ".join([f"{k.capitalize()}: {v:.4f}" for k, v in metrics.items()])
    if prefix:
        logger.info(f"{prefix} Metrics - {metric_str}")
    else:
        logger.info(f"Metrics - {metric_str}")
