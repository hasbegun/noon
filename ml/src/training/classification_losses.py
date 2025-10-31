"""
Loss functions for food classification tasks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing

    Label smoothing helps prevent overfitting by softening the hard targets.
    Instead of [0, 0, 1, 0], we use [ε/K, ε/K, 1-ε+ε/K, ε/K] where K is num_classes.
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1):
        """
        Args:
            num_classes: Number of classes
            smoothing: Smoothing parameter (0.0 = no smoothing, 0.1 = 10% smoothing)
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Model logits (B, num_classes)
            targets: Ground truth labels (B,)

        Returns:
            Loss value
        """
        # Convert logits to log probabilities
        log_probs = F.log_softmax(predictions, dim=1)

        # Create smooth target distribution
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)

        # Calculate loss
        loss = -torch.sum(true_dist * log_probs, dim=1).mean()

        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance

    Focal loss down-weights easy examples and focuses on hard examples.
    FL(p_t) = -α(1-p_t)^γ * log(p_t)
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        """
        Args:
            alpha: Weighting factor (balance between positive/negative)
            gamma: Focusing parameter (0 = CE loss, higher = more focus on hard examples)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Model logits (B, num_classes)
            targets: Ground truth labels (B,)

        Returns:
            Loss value
        """
        # Get probabilities
        probs = F.softmax(predictions, dim=1)

        # Get probability of correct class
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        p_t = torch.gather(probs, 1, targets.unsqueeze(1)).squeeze(1)

        # Calculate focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma

        # Focal loss
        loss = self.alpha * focal_term * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class CombinedClassificationLoss(nn.Module):
    """
    Combined loss for food classification

    Combines:
    1. Cross-Entropy with label smoothing
    2. Focal loss for handling imbalance

    This helps with:
    - Preventing overfitting (label smoothing)
    - Handling class imbalance (focal loss)
    """

    def __init__(
        self,
        num_classes: int,
        ce_weight: float = 0.7,
        focal_weight: float = 0.3,
        label_smoothing: float = 0.1,
        focal_gamma: float = 2.0,
    ):
        """
        Args:
            num_classes: Number of classes
            ce_weight: Weight for cross-entropy loss
            focal_weight: Weight for focal loss
            label_smoothing: Label smoothing factor
            focal_gamma: Focal loss gamma parameter
        """
        super().__init__()

        self.ce_weight = ce_weight
        self.focal_weight = focal_weight

        self.ce_loss = LabelSmoothingCrossEntropy(
            num_classes=num_classes,
            smoothing=label_smoothing
        )

        self.focal_loss = FocalLoss(
            alpha=1.0,
            gamma=focal_gamma,
            reduction='mean'
        )

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Model logits (B, num_classes)
            targets: Ground truth labels (B,)

        Returns:
            Combined loss value
        """
        ce = self.ce_loss(predictions, targets)
        focal = self.focal_loss(predictions, targets)

        total_loss = self.ce_weight * ce + self.focal_weight * focal

        return total_loss


class ClassificationWithNutritionLoss(nn.Module):
    """
    Combined loss for joint classification + nutrition regression

    This is used when training FoodRecognitionWithNutrition model
    on Nutrition5k dataset.
    """

    def __init__(
        self,
        num_classes: int,
        classification_weight: float = 1.0,
        nutrition_weight: float = 0.5,
        label_smoothing: float = 0.1,
    ):
        """
        Args:
            num_classes: Number of food classes
            classification_weight: Weight for classification loss
            nutrition_weight: Weight for nutrition regression loss
            label_smoothing: Label smoothing for classification
        """
        super().__init__()

        self.classification_weight = classification_weight
        self.nutrition_weight = nutrition_weight

        # Classification loss
        self.classification_loss = CombinedClassificationLoss(
            num_classes=num_classes,
            label_smoothing=label_smoothing,
        )

        # Nutrition regression loss (MSE with normalization)
        self.nutrition_loss = nn.MSELoss()

        # Normalization factors for nutrition values (approximate)
        # These help balance the different scales of nutrition values
        self.nutrition_scales = {
            'calories': 500.0,      # ~0-2000 kcal
            'protein_g': 50.0,      # ~0-100g
            'carb_g': 100.0,        # ~0-200g
            'fat_g': 50.0,          # ~0-100g
            'mass_g': 500.0,        # ~0-1000g
        }

    def forward(
        self,
        class_predictions: torch.Tensor,
        nutrition_predictions: torch.Tensor,
        class_targets: torch.Tensor,
        nutrition_targets: torch.Tensor,
    ) -> tuple:
        """
        Args:
            class_predictions: Classification logits (B, num_classes)
            nutrition_predictions: Nutrition predictions (B, 5)
            class_targets: Classification targets (B,)
            nutrition_targets: Nutrition targets (B, 5)

        Returns:
            total_loss, classification_loss, nutrition_loss
        """
        # Classification loss
        cls_loss = self.classification_loss(class_predictions, class_targets)

        # Nutrition regression loss (normalized)
        # Normalize predictions and targets to similar scales
        nutrition_pred_norm = nutrition_predictions / torch.tensor(
            [
                self.nutrition_scales['calories'],
                self.nutrition_scales['protein_g'],
                self.nutrition_scales['carb_g'],
                self.nutrition_scales['fat_g'],
                self.nutrition_scales['mass_g'],
            ],
            device=nutrition_predictions.device
        )

        nutrition_target_norm = nutrition_targets / torch.tensor(
            [
                self.nutrition_scales['calories'],
                self.nutrition_scales['protein_g'],
                self.nutrition_scales['carb_g'],
                self.nutrition_scales['fat_g'],
                self.nutrition_scales['mass_g'],
            ],
            device=nutrition_targets.device
        )

        nutr_loss = self.nutrition_loss(nutrition_pred_norm, nutrition_target_norm)

        # Combined loss
        total_loss = (
            self.classification_weight * cls_loss +
            self.nutrition_weight * nutr_loss
        )

        return total_loss, cls_loss, nutr_loss
