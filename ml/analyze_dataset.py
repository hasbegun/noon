#!/usr/bin/env python3
"""
Analyze dataset to understand mask distribution
Shows how many samples have empty vs. valid segmentation masks
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from config import config
from data_process.dataset import FoodDataset


def analyze_masks(parquet_file: Path, sample_limit: int = None):
    """Analyze mask distribution in dataset"""
    logger.info(f"Loading dataset: {parquet_file}")
    dataset = FoodDataset(parquet_file, mode="train")

    total_samples = len(dataset)
    samples_to_check = min(sample_limit, total_samples) if sample_limit else total_samples

    logger.info(f"Analyzing {samples_to_check} / {total_samples} samples...")

    empty_masks = 0
    valid_masks = 0
    mask_means = []

    for i in tqdm(range(samples_to_check), desc="Analyzing masks"):
        sample = dataset[i]
        mask = sample["mask"]

        mask_mean = mask.float().mean().item()
        mask_means.append(mask_mean)

        if mask_mean < 0.001:  # Essentially empty
            empty_masks += 1
        else:
            valid_masks += 1

    # Statistics
    empty_pct = (empty_masks / samples_to_check) * 100
    valid_pct = (valid_masks / samples_to_check) * 100

    logger.info("=" * 60)
    logger.info("Dataset Mask Analysis Results")
    logger.info("=" * 60)
    logger.info(f"Total samples analyzed: {samples_to_check}")
    logger.info(f"Empty masks (≈0): {empty_masks} ({empty_pct:.1f}%)")
    logger.info(f"Valid masks (>0): {valid_masks} ({valid_pct:.1f}%)")
    logger.info("=" * 60)

    if empty_pct > 50:
        logger.warning("⚠️  More than 50% of samples have empty masks!")
        logger.warning("   This may indicate:")
        logger.warning("   1. Dataset lacks segmentation annotations")
        logger.warning("   2. Mask generation failed for many samples")
        logger.warning("   3. Expected behavior (some images have no food items)")
        logger.warning("")
        logger.warning("   Recommendation:")
        logger.warning("   - If unexpected: Check data preprocessing pipeline")
        logger.warning("   - If expected: Training should still work, but F1 scores may be lower")
    elif empty_pct > 20:
        logger.info(f"ℹ️  {empty_pct:.1f}% empty masks is common for food detection datasets")
        logger.info("   Training should work fine with combined loss function")
    else:
        logger.success(f"✓ Only {empty_pct:.1f}% empty masks - good dataset quality!")

    return {
        "total": samples_to_check,
        "empty": empty_masks,
        "valid": valid_masks,
        "empty_pct": empty_pct,
        "valid_pct": valid_pct,
        "mask_means": mask_means,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze dataset mask distribution")
    parser.add_argument(
        "--data-file",
        type=Path,
        default=config.processed_data_path / "train.parquet",
        help="Path to parquet file",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=1000,
        help="Number of samples to analyze (None for all)",
    )

    args = parser.parse_args()

    if not args.data_file.exists():
        logger.error(f"File not found: {args.data_file}")
        sys.exit(1)

    results = analyze_masks(args.data_file, args.sample_limit)
