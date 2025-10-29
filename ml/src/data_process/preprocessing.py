"""
Data preprocessing pipeline for food detection datasets
Handles multiple dataset formats and ensures data quality
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image
from tqdm import tqdm

from config import config


class DataPreprocessor:
    """Preprocess and validate food image datasets"""

    def __init__(self, raw_data_path: Optional[Path] = None, output_path: Optional[Path] = None):
        self.raw_data_path = raw_data_path or config.raw_data_path
        self.output_path = output_path or config.processed_data_path
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.dataset_processors = {
            "nutrition5k": self._process_nutrition5k,
            "food-101": self._process_food101,
            "UECFOOD100": self._process_uecfood100,
            "ifood2019": self._process_ifood2019,
        }

    def process_all_datasets(self) -> pd.DataFrame:
        """Process all available datasets and create unified dataframe"""
        logger.info("Starting data preprocessing pipeline")

        all_data = []
        for dataset_name, processor in self.dataset_processors.items():
            dataset_path = self.raw_data_path / dataset_name
            if dataset_path.exists():
                logger.info(f"Processing {dataset_name} dataset")
                try:
                    data = processor(dataset_path)
                    all_data.append(data)
                    logger.info(f"Processed {len(data)} samples from {dataset_name}")
                except Exception as e:
                    logger.error(f"Error processing {dataset_name}: {e}")
            else:
                logger.warning(f"Dataset {dataset_name} not found at {dataset_path}")

        if not all_data:
            raise ValueError("No datasets were successfully processed")

        # Combine all datasets
        df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total samples before validation: {len(df)}")

        # Validate and clean data
        df = self._validate_data(df)
        logger.info(f"Total samples after validation: {len(df)}")

        # Save processed data
        output_file = self.output_path / "processed_data.parquet"
        df.to_parquet(output_file, index=False)
        logger.info(f"Saved processed data to {output_file}")

        # Create train/val/test splits
        self._create_splits(df)

        return df

    def _process_nutrition5k(self, dataset_path: Path) -> pd.DataFrame:
        """Process Nutrition5k dataset"""
        data = []
        metadata_dir = dataset_path / "metadata"

        if not metadata_dir.exists():
            logger.warning(f"Metadata directory not found in {dataset_path}")
            return pd.DataFrame()

        # Read dish metadata
        dish_metadata_file = metadata_dir / "dish_metadata_cafe1.csv"
        if dish_metadata_file.exists():
            metadata = pd.read_csv(dish_metadata_file)

            imagery_dir = dataset_path / "imagery"
            for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Nutrition5k"):
                dish_id = row.get("dish_id")
                if pd.isna(dish_id):
                    continue

                # Find images for this dish
                dish_dir = imagery_dir / "realsense_overhead" / str(dish_id)
                if dish_dir.exists():
                    for img_file in dish_dir.glob("*.png"):
                        if self._validate_image(img_file):
                            data.append({
                                "image_path": str(img_file.absolute()),
                                "dataset": "nutrition5k",
                                "dish_id": dish_id,
                                "calories": row.get("total_calories"),
                                "mass_g": row.get("total_mass"),
                                "protein_g": row.get("total_protein"),
                                "carb_g": row.get("total_carb"),
                                "fat_g": row.get("total_fat"),
                                "has_nutrition": True,
                                "has_segmentation": False,
                            })

        return pd.DataFrame(data)

    def _process_food101(self, dataset_path: Path) -> pd.DataFrame:
        """Process Food-101 dataset"""
        data = []
        images_dir = dataset_path / "images"

        if not images_dir.exists():
            logger.warning(f"Images directory not found in {dataset_path}")
            return pd.DataFrame()

        # Food-101 is organized by class folders
        for class_dir in tqdm(list(images_dir.iterdir()), desc="Food-101"):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            for img_file in class_dir.glob("*.jpg"):
                if self._validate_image(img_file):
                    data.append({
                        "image_path": str(img_file.absolute()),
                        "dataset": "food-101",
                        "food_class": class_name,
                        "has_nutrition": False,
                        "has_segmentation": False,
                    })

        return pd.DataFrame(data)

    def _process_uecfood100(self, dataset_path: Path) -> pd.DataFrame:
        """Process UECFOOD100 dataset"""
        data = []

        # Check for category.txt
        category_file = dataset_path / "category.txt"
        if category_file.exists():
            with open(category_file, "r", encoding="utf-8") as f:
                categories = {
                    int(line.split("\t")[0]): line.split("\t")[1].strip()
                    for line in f.readlines()
                    if "\t" in line
                }
        else:
            categories = {}

        # Process images
        for class_id in tqdm(range(1, 101), desc="UECFOOD100"):
            class_dir = dataset_path / str(class_id)
            if not class_dir.exists():
                continue

            class_name = categories.get(class_id, f"class_{class_id}")

            for img_file in class_dir.glob("*.jpg"):
                if self._validate_image(img_file):
                    # Check for bounding box annotation
                    bbox_file = img_file.with_suffix(".txt")
                    has_bbox = bbox_file.exists()

                    data.append({
                        "image_path": str(img_file.absolute()),
                        "dataset": "UECFOOD100",
                        "food_class": class_name,
                        "has_nutrition": False,
                        "has_segmentation": has_bbox,
                        "annotation_path": str(bbox_file.absolute()) if has_bbox else None,
                    })

        return pd.DataFrame(data)

    def _process_ifood2019(self, dataset_path: Path) -> pd.DataFrame:
        """Process iFood2019 dataset"""
        data = []

        # Look for train/val/test splits
        for split in ["train", "val", "test"]:
            split_dir = dataset_path / split
            if not split_dir.exists():
                continue

            # Check for annotations
            ann_file = dataset_path / f"{split}.json"
            if ann_file.exists():
                with open(ann_file, "r") as f:
                    annotations = json.load(f)

                images_info = {img["id"]: img for img in annotations.get("images", [])}
                categories = {cat["id"]: cat["name"] for cat in annotations.get("categories", [])}

                for ann in tqdm(annotations.get("annotations", []), desc=f"iFood2019-{split}"):
                    img_id = ann.get("image_id")
                    img_info = images_info.get(img_id)

                    if img_info:
                        img_path = split_dir / img_info.get("file_name", "")
                        if img_path.exists() and self._validate_image(img_path):
                            data.append({
                                "image_path": str(img_path.absolute()),
                                "dataset": "ifood2019",
                                "split": split,
                                "food_class": categories.get(ann.get("category_id")),
                                "has_nutrition": False,
                                "has_segmentation": "segmentation" in ann,
                            })

        return pd.DataFrame(data)

    def _validate_image(self, img_path: Path) -> bool:
        """Validate that image can be loaded and has valid dimensions"""
        try:
            img = Image.open(img_path)
            width, height = img.size

            # Check minimum dimensions
            if width < 100 or height < 100:
                return False

            # Check if image is corrupted
            img.verify()
            return True
        except Exception:
            return False

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the dataset"""
        initial_count = len(df)

        # Remove rows with missing image paths
        df = df.dropna(subset=["image_path"])

        # Verify all image files exist
        df = df[df["image_path"].apply(lambda x: Path(x).exists())]

        # Remove duplicates
        df = df.drop_duplicates(subset=["image_path"])

        # For nutrition data, remove rows with missing critical values
        nutrition_cols = ["calories", "mass_g", "protein_g", "carb_g", "fat_g"]
        for idx, row in df.iterrows():
            if row.get("has_nutrition", False):
                # Check if any nutrition value is missing or invalid
                if any(pd.isna(row.get(col)) or row.get(col, 0) <= 0 for col in nutrition_cols):
                    df.at[idx, "has_nutrition"] = False

        removed_count = initial_count - len(df)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} invalid samples")

        return df

    def _create_splits(self, df: pd.DataFrame) -> None:
        """Create train/val/test splits"""
        # Shuffle data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Calculate split indices
        n = len(df)
        train_end = int(n * config.train_split)
        val_end = train_end + int(n * config.val_split)

        # Create splits
        train_df = df[:train_end]
        val_df = df[train_end:val_end]
        test_df = df[val_end:]

        # Save splits
        train_df.to_parquet(self.output_path / "train.parquet", index=False)
        val_df.to_parquet(self.output_path / "val.parquet", index=False)
        test_df.to_parquet(self.output_path / "test.parquet", index=False)

        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # Save statistics
        stats = {
            "total_samples": len(df),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "datasets": df["dataset"].value_counts().to_dict(),
            "samples_with_nutrition": int(df["has_nutrition"].sum()),
            "samples_with_segmentation": int(df["has_segmentation"].sum()),
        }

        stats_file = self.output_path / "statistics.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved statistics to {stats_file}")
