"""
Custom PyTorch Dataset for food detection with segmentation
"""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from ..config import config


class FoodDataset(Dataset):
    """Food image dataset with segmentation masks"""

    def __init__(
        self,
        data_file: Path,
        transform: Optional[A.Compose] = None,
        mode: str = "train",
    ):
        """
        Args:
            data_file: Path to parquet file with image metadata
            transform: Albumentations transforms
            mode: 'train', 'val', or 'test'
        """
        self.df = pd.read_parquet(data_file)
        self.transform = transform or self._get_default_transform(mode)
        self.mode = mode

        # Filter valid samples
        self.df = self.df[self.df["image_path"].apply(lambda x: Path(x).exists())]

        # Create class to index mapping
        if "food_class" in self.df.columns:
            self.classes = sorted(self.df["food_class"].dropna().unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        else:
            self.classes = []
            self.class_to_idx = {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset"""
        row = self.df.iloc[idx]

        # Load image
        image_path = Path(row["image_path"])
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Prepare sample dict
        sample = {
            "image": image,
            "image_path": str(image_path),
            "dataset": row.get("dataset", "unknown"),
        }

        # Add segmentation mask if available
        if row.get("has_segmentation", False) and pd.notna(row.get("annotation_path")):
            mask = self._load_mask(Path(row["annotation_path"]), image.shape[:2])
            sample["mask"] = mask
        else:
            # Create empty mask
            sample["mask"] = np.zeros(image.shape[:2], dtype=np.uint8)

        # Add class label if available
        if "food_class" in row and pd.notna(row["food_class"]):
            sample["label"] = self.class_to_idx.get(row["food_class"], -1)
            sample["class_name"] = row["food_class"]
        else:
            sample["label"] = -1
            sample["class_name"] = "unknown"

        # Add nutrition info if available
        if row.get("has_nutrition", False):
            sample["nutrition"] = {
                "calories": row.get("calories", 0.0),
                "mass_g": row.get("mass_g", 0.0),
                "protein_g": row.get("protein_g", 0.0),
                "carb_g": row.get("carb_g", 0.0),
                "fat_g": row.get("fat_g", 0.0),
            }
        else:
            sample["nutrition"] = None

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=sample["image"], mask=sample["mask"])
            sample["image"] = transformed["image"]
            sample["mask"] = transformed["mask"]

        # Convert mask to tensor if not already
        if not isinstance(sample["mask"], torch.Tensor):
            sample["mask"] = torch.from_numpy(sample["mask"]).long()

        return sample

    def _load_mask(self, annotation_path: Path, image_shape: Tuple[int, int]) -> np.ndarray:
        """Load segmentation mask from annotation file"""
        height, width = image_shape
        mask = np.zeros((height, width), dtype=np.uint8)

        if annotation_path.suffix == ".txt":
            # UECFOOD format: bounding boxes
            try:
                with open(annotation_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            x1, y1, x2, y2 = map(int, parts[:4])
                            mask[y1:y2, x1:x2] = 1
            except Exception:
                pass
        elif annotation_path.suffix == ".png":
            # Binary mask
            try:
                mask = cv2.imread(str(annotation_path), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (width, height))
                mask = (mask > 0).astype(np.uint8)
            except Exception:
                pass

        return mask

    def _get_default_transform(self, mode: str) -> A.Compose:
        """Get default augmentation transforms"""
        if mode == "train" and config.use_augmentation:
            return A.Compose([
                A.Resize(config.image_size, config.image_size),
                A.HorizontalFlip(p=config.augmentation_prob),
                A.RandomBrightnessContrast(p=config.augmentation_prob),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=15,
                    p=config.augmentation_prob
                ),
                A.OneOf([
                    A.GaussNoise(p=1.0),
                    A.GaussianBlur(p=1.0),
                    A.MotionBlur(p=1.0),
                ], p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(config.image_size, config.image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ])

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets"""
        if not self.class_to_idx:
            return torch.ones(1)

        class_counts = self.df["food_class"].value_counts()
        weights = 1.0 / class_counts.values
        weights = weights / weights.sum()

        return torch.FloatTensor(weights)
