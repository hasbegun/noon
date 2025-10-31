"""
Custom loss functions for food segmentation training
Addresses the issue of loss going to 0.0 by preventing model collapse
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    Better for class imbalance than BCE alone
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Logits from model (B, 1, H, W)
            targets: Ground truth masks (B, 1, H, W) or (B, H, W)
        """
        # Apply sigmoid to get probabilities
        predictions = torch.sigmoid(predictions)

        # Ensure targets have same shape as predictions
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1).float()

        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()

        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)

        # Return loss (1 - dice)
        return 1.0 - dice_score


class FocalLoss(nn.Module):
    """
    Focal Loss to focus on hard examples
    Helps with class imbalance (many background pixels, few foreground)

    alpha: Weight for positive class (higher = more focus on positive class)
           Default 0.75 gives 3x more weight to food pixels vs background
    gamma: Focusing parameter (higher = more focus on hard examples)
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Logits from model (B, 1, H, W)
            targets: Ground truth masks (B, 1, H, W) or (B, H, W)
        """
        # Ensure targets have same shape
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        targets = targets.float()

        # Calculate BCE loss without reduction
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none'
        )

        # Calculate pt (probability of correct class)
        pt = torch.exp(-bce_loss)

        # Apply alpha weighting: alpha for positive class, (1-alpha) for negative class
        # This gives more weight to the minority class (food pixels)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Calculate focal loss with proper alpha weighting
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class IoULoss(nn.Module):
    """
    IoU (Intersection over Union) Loss
    Directly optimizes for IoU metric
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Logits from model (B, 1, H, W)
            targets: Ground truth masks (B, 1, H, W) or (B, H, W)
        """
        # Apply sigmoid
        predictions = torch.sigmoid(predictions)

        # Ensure targets have same shape
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        targets = targets.float()

        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        total = predictions.sum() + targets.sum()
        union = total - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)

        return 1.0 - iou


class CombinedSegmentationLoss(nn.Module):
    """
    Combined loss for robust segmentation training
    Prevents model from collapsing to all-zero predictions

    Components:
    1. BCE Loss: Pixel-wise classification
    2. Dice Loss: Overlap maximization (good for imbalance)
    3. Focal Loss: Focus on hard examples

    This combination:
    - Prevents trivial all-zero solutions
    - Handles class imbalance (many background pixels)
    - Focuses on hard-to-segment regions
    """
    def __init__(
        self,
        bce_weight=0.4,
        dice_weight=0.4,
        focal_weight=0.2,
        log_components=False,
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.log_components = log_components

        # Validate weights sum to 1.0
        total_weight = bce_weight + dice_weight + focal_weight
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                f"Loss weights sum to {total_weight:.3f}, not 1.0. "
                f"This is OK but verify it's intentional."
            )

        logger.info(
            f"CombinedSegmentationLoss initialized: "
            f"BCE={bce_weight}, Dice={dice_weight}, Focal={focal_weight}"
        )

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Logits from model (B, 1, H, W)
            targets: Ground truth masks (B, 1, H, W) or (B, H, W)
        """
        # Calculate individual losses
        bce_loss = self.bce(predictions, targets.unsqueeze(1).float() if targets.dim() == 3 else targets.float())
        dice_loss = self.dice(predictions, targets)
        focal_loss = self.focal(predictions, targets)

        # Combined loss
        total_loss = (
            self.bce_weight * bce_loss +
            self.dice_weight * dice_loss +
            self.focal_weight * focal_loss
        )

        # Optionally log components (for debugging)
        if self.log_components and torch.rand(1).item() < 0.01:  # Log 1% of the time
            logger.debug(
                f"Loss components: BCE={bce_loss:.4f}, "
                f"Dice={dice_loss:.4f}, Focal={focal_loss:.4f}, "
                f"Total={total_loss:.4f}"
            )

        return total_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice loss
    Useful when false positives and false negatives have different costs
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super().__init__()
        self.alpha = alpha  # Weight for false positives
        self.beta = beta    # Weight for false negatives
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)

        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # True positives, false positives, false negatives
        TP = (predictions * targets).sum()
        FP = (predictions * (1 - targets)).sum()
        FN = ((1 - predictions) * targets).sum()

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        return 1.0 - tversky


def get_loss_function(loss_type='combined'):
    """
    Factory function to get loss function by name

    Args:
        loss_type: One of 'bce', 'dice', 'focal', 'iou', 'combined', 'tversky'

    Returns:
        Loss function instance
    """
    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'dice':
        return DiceLoss()
    elif loss_type == 'focal':
        return FocalLoss()
    elif loss_type == 'iou':
        return IoULoss()
    elif loss_type == 'combined':
        return CombinedSegmentationLoss()
    elif loss_type == 'tversky':
        return TverskyLoss()
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Choose from: bce, dice, focal, iou, combined, tversky"
        )
