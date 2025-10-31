"""
Mixup and CutMix data augmentation for improved generalization

References:
- Mixup: https://arxiv.org/abs/1710.09412
- CutMix: https://arxiv.org/abs/1905.04899
"""

import numpy as np
import torch
import torch.nn.functional as F


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """
    Mixup data augmentation - linearly interpolate between two samples

    Args:
        x: Input images (B, C, H, W)
        y: Labels (B,)
        alpha: Mixup interpolation strength (beta distribution parameter)

    Returns:
        mixed_x: Mixed images
        y_a: First set of labels
        y_b: Second set of labels
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    if batch_size == 0:
        return x, y, y, 1.0

    # Random permutation
    index = torch.randperm(batch_size).to(x.device)

    # Mix images: mixed = lam * image_a + (1 - lam) * image_b
    mixed_x = lam * x + (1 - lam) * x[index]

    y_a = y
    y_b = y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Mixup loss function

    Args:
        criterion: Loss function
        pred: Model predictions
        y_a: First set of labels
        y_b: Second set of labels
        lam: Mixing coefficient

    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def rand_bbox(size, lam):
    """
    Generate random bounding box for CutMix

    Args:
        size: Image size (B, C, H, W)
        lam: Mixing coefficient

    Returns:
        Bounding box coordinates (x1, y1, x2, y2)
    """
    W = size[2]
    H = size[3]

    # Calculate cut size
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform random center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Bounding box
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """
    CutMix data augmentation - cut and paste patches between samples

    Args:
        x: Input images (B, C, H, W)
        y: Labels (B,)
        alpha: CutMix interpolation strength (beta distribution parameter)

    Returns:
        mixed_x: Mixed images
        y_a: First set of labels
        y_b: Second set of labels
        lam: Actual mixing ratio based on box size
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    if batch_size == 0:
        return x, y, y, 1.0

    # Random permutation
    rand_index = torch.randperm(batch_size).to(x.device)

    y_a = y
    y_b = y[rand_index]

    # Generate random bounding box
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)

    # Cut and paste
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda to exactly match the box area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    return x, y_a, y_b, lam


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """
    CutMix loss function (same as mixup)

    Args:
        criterion: Loss function
        pred: Model predictions
        y_a: First set of labels
        y_b: Second set of labels
        lam: Mixing coefficient

    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class MixupCutmixAugmentation:
    """
    Combined Mixup and CutMix augmentation

    Randomly applies either Mixup or CutMix to each batch
    """

    def __init__(
        self,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        mixup_prob: float = 0.5,
        cutmix_prob: float = 0.5,
    ):
        """
        Initialize combined augmentation

        Args:
            mixup_alpha: Mixup strength
            cutmix_alpha: CutMix strength
            mixup_prob: Probability of applying mixup
            cutmix_prob: Probability of applying cutmix
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        """
        Apply mixup or cutmix randomly

        Args:
            x: Input images
            y: Labels

        Returns:
            Augmented images and labels
        """
        # Decide which augmentation to use
        use_mixup = np.random.rand() < self.mixup_prob
        use_cutmix = np.random.rand() < self.cutmix_prob

        if use_mixup and use_cutmix:
            # Both: choose one randomly
            if np.random.rand() < 0.5:
                return mixup_data(x, y, self.mixup_alpha)
            else:
                return cutmix_data(x, y, self.cutmix_alpha)
        elif use_mixup:
            return mixup_data(x, y, self.mixup_alpha)
        elif use_cutmix:
            return cutmix_data(x, y, self.cutmix_alpha)
        else:
            # No augmentation
            return x, y, y, 1.0

    def compute_loss(self, criterion, pred, y_a, y_b, lam):
        """
        Compute mixed loss

        Args:
            criterion: Loss function
            pred: Model predictions
            y_a: First set of labels
            y_b: Second set of labels
            lam: Mixing coefficient

        Returns:
            Mixed loss
        """
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
