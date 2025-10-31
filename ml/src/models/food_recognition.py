"""
Food Recognition Model for classifying detected food items

This model takes image crops from SAM2 segmentation and classifies them
into food categories. It's designed to work with Nutrition5k and Food-101 datasets.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from loguru import logger


class FoodRecognitionModel(nn.Module):
    """
    Food recognition model using pretrained backbone + classification head

    This model is designed to classify food items from image crops.
    It uses a pretrained backbone (EfficientNet-B0) for feature extraction
    and a custom head for food classification.
    """

    def __init__(
        self,
        num_classes: int = 101,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        """
        Initialize food recognition model

        Args:
            num_classes: Number of food categories to classify
            backbone: Backbone architecture (efficientnet_b0, resnet50, mobilenet_v3_small)
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout probability for regularization
        """
        super().__init__()

        self.num_classes = num_classes
        self.backbone_name = backbone

        # Create backbone
        if backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            feature_dim = 1280  # EfficientNet-B0 output dimension
            # Remove the original classifier
            self.backbone.classifier = nn.Identity()

        elif backbone == "efficientnet_b3":
            weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b3(weights=weights)
            feature_dim = 1536  # EfficientNet-B3 output dimension
            self.backbone.classifier = nn.Identity()

        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            feature_dim = 2048  # ResNet50 output dimension
            # Remove the original FC layer
            self.backbone.fc = nn.Identity()

        elif backbone == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.mobilenet_v3_small(weights=weights)
            feature_dim = 576  # MobileNetV3-Small output dimension
            self.backbone.classifier = nn.Identity()

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

        # Initialize classifier weights
        self._init_classifier()

        logger.info(f"Created FoodRecognitionModel: {backbone}, {num_classes} classes")

    def _init_classifier(self):
        """Initialize classifier head with proper weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (B, 3, H, W) - RGB images

        Returns:
            Logits tensor (B, num_classes)
        """
        # Extract features
        features = self.backbone(x)

        # Classify
        logits = self.classifier(features)

        return logits

    def predict(self, x: torch.Tensor, return_probabilities: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with confidence scores

        Args:
            x: Input tensor (B, 3, H, W)
            return_probabilities: Return probabilities instead of logits

        Returns:
            predictions: Class indices (B,)
            confidences: Confidence scores (B,)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)

            if return_probabilities:
                probs = F.softmax(logits, dim=1)
                confidences, predictions = torch.max(probs, dim=1)
            else:
                confidences, predictions = torch.max(logits, dim=1)

        return predictions, confidences

    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen")

    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: Path):
        """Save model checkpoint"""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_classes': self.num_classes,
            'backbone': self.backbone_name,
        }, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "FoodRecognitionModel":
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=device)

        model = cls(
            num_classes=checkpoint['num_classes'],
            backbone=checkpoint['backbone'],
            pretrained=False,  # Don't load ImageNet weights
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")

        return model


class FoodRecognitionWithNutrition(nn.Module):
    """
    Extended recognition model that also predicts nutrition values

    This is useful for Nutrition5k dataset which has ground-truth nutrition.
    The model learns to predict both food category and nutrition values jointly.
    """

    def __init__(
        self,
        num_classes: int = 101,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        """
        Initialize recognition + nutrition model

        Args:
            num_classes: Number of food categories
            backbone: Backbone architecture
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout probability
        """
        super().__init__()

        # Create base recognition model
        self.recognition_model = FoodRecognitionModel(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            dropout=dropout,
        )

        # Get feature dimension from backbone
        if backbone.startswith("efficientnet_b0"):
            feature_dim = 1280
        elif backbone.startswith("efficientnet_b3"):
            feature_dim = 1536
        elif backbone == "resnet50":
            feature_dim = 2048
        elif backbone == "mobilenet_v3_small":
            feature_dim = 576
        else:
            feature_dim = 1280

        # Nutrition regression head (predicts: calories, protein, carbs, fat, mass)
        self.nutrition_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 5),  # 5 nutrition values
            nn.ReLU(),  # All nutrition values are positive
        )

        self._init_nutrition_head()

        logger.info(f"Created FoodRecognitionWithNutrition: {backbone}, {num_classes} classes + nutrition")

    def _init_nutrition_head(self):
        """Initialize nutrition head"""
        for m in self.nutrition_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            class_logits: Classification logits (B, num_classes)
            nutrition_pred: Nutrition predictions (B, 5) - [calories, protein, carbs, fat, mass]
        """
        # Extract features
        features = self.recognition_model.backbone(x)

        # Classification
        class_logits = self.recognition_model.classifier(features)

        # Nutrition regression
        nutrition_pred = self.nutrition_head(features)

        return class_logits, nutrition_pred

    def predict(
        self,
        x: torch.Tensor,
        return_probabilities: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Make predictions

        Args:
            x: Input tensor (B, 3, H, W)
            return_probabilities: Return probabilities instead of logits

        Returns:
            predictions: Class indices (B,)
            confidences: Confidence scores (B,)
            nutrition: Dict with nutrition values
        """
        self.eval()
        with torch.no_grad():
            class_logits, nutrition_pred = self.forward(x)

            if return_probabilities:
                probs = F.softmax(class_logits, dim=1)
                confidences, predictions = torch.max(probs, dim=1)
            else:
                confidences, predictions = torch.max(class_logits, dim=1)

            nutrition = {
                'calories': nutrition_pred[:, 0],
                'protein_g': nutrition_pred[:, 1],
                'carb_g': nutrition_pred[:, 2],
                'fat_g': nutrition_pred[:, 3],
                'mass_g': nutrition_pred[:, 4],
            }

        return predictions, confidences, nutrition
