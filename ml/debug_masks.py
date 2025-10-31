#!/usr/bin/env python3
"""
Debug script to check if training masks are valid (not all zeros)
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import numpy as np
from data_process import create_data_loaders
from config import config
from loguru import logger

def check_masks():
    """Check if masks in training data are valid"""

    # Create data loaders
    logger.info("Creating data loaders...")
    data_loaders = create_data_loaders(
        train_file=config.processed_data_path / "train.parquet",
        val_file=config.processed_data_path / "val.parquet",
        batch_size=8,
        num_workers=0,  # Use 0 for debugging
    )

    if "train" not in data_loaders:
        logger.error("No training data found!")
        return

    train_loader = data_loaders["train"]
    logger.info(f"Train loader has {len(train_loader)} batches")

    # Check first few batches
    zero_mask_count = 0
    non_zero_mask_count = 0
    total_samples = 0

    logger.info("Checking first 10 batches...")
    for i, batch in enumerate(train_loader):
        if i >= 10:  # Check first 10 batches
            break

        masks = batch["masks"]
        batch_size = masks.shape[0]
        total_samples += batch_size

        logger.info(f"\nBatch {i+1}:")
        logger.info(f"  Masks shape: {masks.shape}")
        logger.info(f"  Masks dtype: {masks.dtype}")

        # Check each mask in batch
        for j in range(batch_size):
            mask = masks[j]
            mask_sum = mask.sum().item()
            mask_max = mask.max().item()
            mask_min = mask.min().item()

            if mask_sum == 0:
                zero_mask_count += 1
                logger.warning(f"    Sample {j}: ALL ZEROS (sum={mask_sum}, min={mask_min}, max={mask_max})")
            else:
                non_zero_mask_count += 1
                logger.info(f"    Sample {j}: sum={mask_sum}, min={mask_min}, max={mask_max}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Summary:")
    logger.info(f"  Total samples checked: {total_samples}")
    logger.info(f"  Masks with zeros only: {zero_mask_count} ({zero_mask_count/total_samples*100:.1f}%)")
    logger.info(f"  Masks with non-zero values: {non_zero_mask_count} ({non_zero_mask_count/total_samples*100:.1f}%)")
    logger.info(f"{'='*60}")

    if zero_mask_count == total_samples:
        logger.error("\n❌ PROBLEM FOUND: ALL MASKS ARE ZEROS!")
        logger.error("This is why loss goes to 0.0 - the model learns to output all zeros")
        logger.error("\nPossible causes:")
        logger.error("1. Segmentation annotations are missing or not loaded")
        logger.error("2. Annotation file paths are incorrect")
        logger.error("3. Mask loading logic has a bug")
        logger.error("\nCheck:")
        logger.error("  - data/processed/train.parquet: 'has_segmentation' column")
        logger.error("  - data/processed/train.parquet: 'annotation_path' column")
        logger.error("  - src/data_process/dataset.py: _load_mask() method")
    elif zero_mask_count > total_samples * 0.5:
        logger.warning(f"\n⚠️  WARNING: {zero_mask_count/total_samples*100:.1f}% of masks are all zeros")
        logger.warning("This could cause training issues")
    else:
        logger.info("\n✓ Masks look OK - mix of zero and non-zero masks")

if __name__ == "__main__":
    check_masks()
