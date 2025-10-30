#!/usr/bin/env python3
"""
Debug script to check masks and detect data issues

Usage:
    From ml directory: python -m src.utils.debug_masks
    Or: python src/utils/debug_masks.py
"""
import sys
from pathlib import Path

# Add project root to path
ml_dir = Path(__file__).parent.parent.parent
src_dir = ml_dir / "src"
sys.path.insert(0, str(ml_dir))
sys.path.insert(0, str(src_dir))

import torch
import numpy as np
from config import config
from data_process.loader import create_data_loaders

def analyze_dataset():
    """Analyze masks to detect issues"""
    print("=" * 60)
    print("MASK ANALYSIS - Checking for Data Issues")
    print("=" * 60)

    # Load datasets
    print("\nLoading datasets...")
    loaders = create_data_loaders(
        train_file=config.processed_data_path / "train.parquet",
        val_file=config.processed_data_path / "val.parquet",
        batch_size=8,
        num_workers=0,  # Use 0 for debugging
    )

    train_loader = loaders['train']
    val_loader = loaders.get('val')

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Analyze first 100 batches
    print("\n" + "=" * 60)
    print("TRAINING DATA ANALYSIS")
    print("=" * 60)

    analyze_batches(train_loader, "Training", num_batches=100)

    if val_loader:
        print("\n" + "=" * 60)
        print("VALIDATION DATA ANALYSIS")
        print("=" * 60)
        analyze_batches(val_loader, "Validation", num_batches=100)

def analyze_batches(loader, name, num_batches=100):
    """Analyze batches for issues"""

    mask_stats = {
        'all_zeros': 0,
        'all_ones': 0,
        'partial': 0,
        'total_batches': 0,
        'avg_positive_ratio': []
    }

    print(f"\nAnalyzing {name} data (first {num_batches} batches)...")

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break

        images = batch['images']
        masks = batch['masks']

        # Check shapes
        if i == 0:
            print(f"\nBatch shapes:")
            print(f"  Images: {images.shape}")
            print(f"  Masks: {masks.shape}")
            print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
            print(f"  Mask unique values: {torch.unique(masks).tolist()}")

        # Analyze masks
        for j in range(masks.shape[0]):
            mask = masks[j]
            total_pixels = mask.numel()
            positive_pixels = (mask > 0).sum().item()
            positive_ratio = positive_pixels / total_pixels

            mask_stats['avg_positive_ratio'].append(positive_ratio)

            if positive_ratio == 0:
                mask_stats['all_zeros'] += 1
            elif positive_ratio == 1:
                mask_stats['all_ones'] += 1
            else:
                mask_stats['partial'] += 1

        mask_stats['total_batches'] += 1

    # Calculate statistics
    total_samples = mask_stats['all_zeros'] + mask_stats['all_ones'] + mask_stats['partial']
    avg_positive = np.mean(mask_stats['avg_positive_ratio'])

    print(f"\n{name} Mask Statistics:")
    print(f"  Total samples analyzed: {total_samples}")
    print(f"  All zeros (no food): {mask_stats['all_zeros']} ({mask_stats['all_zeros']/total_samples*100:.1f}%)")
    print(f"  All ones (full food): {mask_stats['all_ones']} ({mask_stats['all_ones']/total_samples*100:.1f}%)")
    print(f"  Partial masks: {mask_stats['partial']} ({mask_stats['partial']/total_samples*100:.1f}%)")
    print(f"  Average positive ratio: {avg_positive:.4f} ({avg_positive*100:.2f}%)")

    # Detect issues
    print(f"\n{name} Issues Detected:")

    issues_found = False

    if mask_stats['all_zeros'] / total_samples > 0.5:
        print(f"  ⚠️  WARNING: {mask_stats['all_zeros']/total_samples*100:.1f}% of masks are all zeros!")
        print(f"     This means most images have no food labels.")
        print(f"     Model will learn to predict all zeros for low loss.")
        issues_found = True

    if avg_positive < 0.01:
        print(f"  ⚠️  WARNING: Only {avg_positive*100:.2f}% of pixels are positive!")
        print(f"     Severe class imbalance - model will predict all zeros.")
        issues_found = True

    if mask_stats['all_ones'] / total_samples > 0.1:
        print(f"  ⚠️  WARNING: {mask_stats['all_ones']/total_samples*100:.1f}% of masks are all ones!")
        print(f"     This is suspicious - masks should have food regions, not full images.")
        issues_found = True

    if not issues_found:
        print(f"  ✓ No major issues detected")

    return mask_stats, issues_found

if __name__ == "__main__":
    try:
        analyze_dataset()

        print("\n" + "=" * 60)
        print("RECOMMENDATIONS")
        print("=" * 60)
        print("\nIf masks are mostly zeros:")
        print("  1. Use weighted loss (BCEWithLogitsLoss with pos_weight)")
        print("  2. Filter out zero-mask samples during training")
        print("  3. Add Dice loss or IoU loss")
        print("  4. Check data preprocessing - masks might be incorrect")
        print("\nIf model predicts all zeros:")
        print("  1. Check model is actually training (gradients flowing)")
        print("  2. Verify masks are correct (not all black)")
        print("  3. Use balanced sampling")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
