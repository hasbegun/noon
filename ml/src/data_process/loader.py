"""
Data loader with multi-node distributed training support
"""
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from config import config
from data_process.dataset import FoodDataset


def create_data_loaders(
    train_file: Optional[Path] = None,
    val_file: Optional[Path] = None,
    test_file: Optional[Path] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing

    Args:
        train_file: Path to training data parquet
        val_file: Path to validation data parquet
        test_file: Path to test data parquet
        batch_size: Batch size per device
        num_workers: Number of data loading workers
        distributed: Whether to use distributed sampling
        rank: Process rank for distributed training
        world_size: Total number of processes

    Returns:
        Dictionary of data loaders
    """
    batch_size = batch_size or config.batch_size
    num_workers = num_workers or config.num_workers

    loaders = {}

    # Training loader
    if train_file and train_file.exists():
        train_dataset = FoodDataset(train_file, mode="train")

        if distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )
            shuffle = False
        else:
            train_sampler = RandomSampler(train_dataset)
            shuffle = False  # Sampler handles shuffling

        loaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=num_workers > 0,
            collate_fn=collate_fn,
        )

    # Validation loader
    if val_file and val_file.exists():
        val_dataset = FoodDataset(val_file, mode="val")

        if distributed:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
            )
        else:
            val_sampler = SequentialSampler(val_dataset)

        loaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            collate_fn=collate_fn,
        )

    # Test loader
    if test_file and test_file.exists():
        test_dataset = FoodDataset(test_file, mode="test")

        if distributed:
            test_sampler = DistributedSampler(
                test_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
            )
        else:
            test_sampler = SequentialSampler(test_dataset)

        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            collate_fn=collate_fn,
        )

    return loaders


def collate_fn(batch):
    """
    Custom collate function to handle variable-sized data
    """
    images = torch.stack([item["image"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])

    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    # Handle nutrition data
    has_nutrition = [item["nutrition"] is not None for item in batch]
    if any(has_nutrition):
        nutrition_data = {
            "calories": [],
            "mass_g": [],
            "protein_g": [],
            "carb_g": [],
            "fat_g": [],
        }
        for item in batch:
            if item["nutrition"]:
                for key in nutrition_data:
                    nutrition_data[key].append(item["nutrition"][key])
            else:
                for key in nutrition_data:
                    nutrition_data[key].append(0.0)

        nutrition_tensors = {
            key: torch.tensor(vals, dtype=torch.float32)
            for key, vals in nutrition_data.items()
        }
    else:
        nutrition_tensors = None

    return {
        "images": images,
        "masks": masks,
        "labels": labels,
        "nutrition": nutrition_tensors,
        "image_paths": [item["image_path"] for item in batch],
        "class_names": [item["class_name"] for item in batch],
        "datasets": [item["dataset"] for item in batch],
    }
