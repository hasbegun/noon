"""
PyTorch Dataset for food classification (recognition) task

This dataset is designed for training food recognition models.
Unlike segmentation datasets, it only needs images and class labels, not masks.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from loguru import logger
from torch.utils.data import Dataset

from config import config
from data_process.food_labels import FoodLabelManager


class FoodClassificationDataset(Dataset):
    """
    Food classification dataset for training recognition models

    This dataset loads images and their corresponding food category labels.
    It's designed for Nutrition5k and Food-101 datasets.
    """

    def __init__(
        self,
        data_file: Path,
        label_manager: Optional[FoodLabelManager] = None,
        transform: Optional[A.Compose] = None,
        mode: str = "train",
        include_nutrition: bool = False,
    ):
        """
        Args:
            data_file: Path to parquet file with image metadata
            label_manager: Label manager for food categories
            transform: Albumentations transforms
            mode: 'train', 'val', or 'test'
            include_nutrition: Include nutrition data (for Nutrition5k)
        """
        self.df = pd.read_parquet(data_file)
        self.mode = mode
        self.include_nutrition = include_nutrition
        self.transform = transform or self._get_default_transform(mode)

        # Setup label manager
        if label_manager is None:
            # Auto-detect from dataset column
            dataset_name = self.df['dataset'].iloc[0] if 'dataset' in self.df.columns else 'food-101'
            if 'nutrition5k' in dataset_name:
                dataset_name = 'nutrition5k'
            elif 'food-101' in dataset_name or 'food101' in dataset_name:
                dataset_name = 'food-101'
            self.label_manager = FoodLabelManager(dataset_name)
        else:
            self.label_manager = label_manager

        # Filter valid samples
        logger.info(f"Loading dataset from {data_file.name}...")
        logger.info(f"Initial samples: {len(self.df)}")

        # Keep only samples with valid image paths and food classes
        valid_mask = (
            self.df["image_path"].apply(self._is_valid_image) &
            self.df["food_class"].notna()
        )
        self.df = self.df[valid_mask].reset_index(drop=True)

        logger.info(f"Valid samples after filtering: {len(self.df)}")

        # Get unique classes in this dataset
        self.unique_classes = sorted(self.df["food_class"].unique())
        logger.info(f"Found {len(self.unique_classes)} unique food classes")

        # Calculate class distribution
        class_counts = self.df["food_class"].value_counts()
        logger.info(f"Top 5 classes: {class_counts.head().to_dict()}")

    def __len__(self) -> int:
        return len(self.df)

    def _is_valid_image(self, image_path_str: str) -> bool:
        """Check if an image file exists and is valid"""
        try:
            image_path = Path(image_path_str)
            return image_path.exists() and image_path.stat().st_size > 0
        except Exception:
            return False

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset"""
        try:
            row = self.df.iloc[idx]

            # Load image
            image_path = Path(row["image_path"])
            image = cv2.imread(str(image_path))

            if image is None:
                logger.warning(f"Failed to load image: {image_path}, returning dummy sample")
                return self._create_dummy_sample()

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Get class label
            class_name = row["food_class"]
            class_idx = self.label_manager.get_class_idx(class_name)

            # Handle unknown classes (assign to last class or skip)
            if class_idx == -1:
                logger.warning(f"Unknown class '{class_name}', using index 0")
                class_idx = 0

            # Prepare sample
            sample = {
                "image": image,
                "image_path": str(image_path),
                "label": class_idx,
                "class_name": class_name,
                "dataset": row.get("dataset", "unknown"),
            }

            # Add nutrition info if requested and available
            if self.include_nutrition and row.get("has_nutrition", False):
                sample["nutrition"] = {
                    "calories": float(row.get("calories", 0.0)),
                    "protein_g": float(row.get("protein_g", 0.0)),
                    "carb_g": float(row.get("carb_g", 0.0)),
                    "fat_g": float(row.get("fat_g", 0.0)),
                    "mass_g": float(row.get("mass_g", 0.0)),
                }
            else:
                sample["nutrition"] = None

            # Apply transforms
            if self.transform:
                try:
                    transformed = self.transform(image=sample["image"])
                    sample["image"] = transformed["image"]
                except Exception as e:
                    logger.error(f"Transform failed for {image_path}: {e}")
                    return self._create_dummy_sample()

            return sample

        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            return self._create_dummy_sample()

    def _create_dummy_sample(self) -> Dict[str, Any]:
        """Create a dummy sample when loading fails"""
        # Create a black image
        dummy_image = np.zeros((config.image_size, config.image_size, 3), dtype=np.uint8)

        # Apply transforms to get proper tensor format
        if self.transform:
            transformed = self.transform(image=dummy_image)
            dummy_image = transformed["image"]
        else:
            dummy_image = torch.from_numpy(dummy_image).permute(2, 0, 1).float()

        return {
            "image": dummy_image,
            "label": 0,  # Default class
            "class_name": "unknown",
            "image_path": "dummy",
            "dataset": "dummy",
            "nutrition": None,
        }

    def _get_default_transform(self, mode: str) -> A.Compose:
        """Get default augmentation transforms for classification"""
        # For classification, we use more aggressive augmentation
        if mode == "train" and config.use_augmentation:
            return A.Compose([
                # Resize with some variation
                A.RandomResizedCrop(
                    height=config.image_size,
                    width=config.image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    p=1.0
                ),
                # Geometric transforms
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=20,
                    p=0.5
                ),
                # Color augmentation
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.5
                ),
                # One of blur/noise
                A.OneOf([
                    A.GaussNoise(p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(p=1.0),
                ], p=0.3),
                # Cutout for regularization
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    p=0.3
                ),
                # Normalize
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ])
        else:
            # Validation/test: just resize and normalize
            return A.Compose([
                A.Resize(config.image_size, config.image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ])

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalanced datasets"""
        class_counts = self.df["food_class"].value_counts()

        # Create weight for each class (inverse frequency)
        weights = torch.zeros(self.label_manager.num_classes)

        for class_name, count in class_counts.items():
            class_idx = self.label_manager.get_class_idx(class_name)
            if class_idx != -1:
                weights[class_idx] = 1.0 / count

        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum() * len(weights)
        else:
            weights = torch.ones(self.label_manager.num_classes)

        return weights

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        stats = {
            "total_samples": len(self.df),
            "num_classes": len(self.unique_classes),
            "classes": self.unique_classes,
            "class_distribution": self.df["food_class"].value_counts().to_dict(),
        }

        if self.include_nutrition and "calories" in self.df.columns:
            nutrition_df = self.df[self.df["has_nutrition"] == True]
            if len(nutrition_df) > 0:
                stats["nutrition_samples"] = len(nutrition_df)
                stats["avg_calories"] = float(nutrition_df["calories"].mean())
                stats["avg_mass_g"] = float(nutrition_df["mass_g"].mean())

        return stats


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for classification batches

    Args:
        batch: List of samples from dataset

    Returns:
        Batched tensors
    """
    # Stack images
    images = torch.stack([item["image"] for item in batch])

    # Stack labels
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    # Collect metadata
    class_names = [item["class_name"] for item in batch]
    image_paths = [item["image_path"] for item in batch]

    result = {
        "image": images,
        "label": labels,
        "class_name": class_names,
        "image_path": image_paths,
    }

    # Include nutrition if available
    if batch[0]["nutrition"] is not None:
        nutrition_data = {
            "calories": [],
            "protein_g": [],
            "carb_g": [],
            "fat_g": [],
            "mass_g": [],
        }

        for item in batch:
            if item["nutrition"] is not None:
                for key in nutrition_data:
                    nutrition_data[key].append(item["nutrition"][key])
            else:
                for key in nutrition_data:
                    nutrition_data[key].append(0.0)

        # Convert to tensors
        for key in nutrition_data:
            nutrition_data[key] = torch.tensor(nutrition_data[key], dtype=torch.float32)

        result["nutrition"] = nutrition_data

    return result
