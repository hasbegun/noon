#!/usr/bin/env python3
"""
Test script to debug dataset loading

Run from ml directory:
    python tests/test_dataset.py
"""
import sys
from pathlib import Path

# Add ml directory to path
ml_dir = Path(__file__).parent.parent
sys.path.insert(0, str(ml_dir))

from src.data_process.dataset import FoodDataset

def test_dataset_loading():
    """Test loading samples from the dataset"""
    # Use relative path from ml directory
    train_file = ml_dir / "data" / "processed" / "train.parquet"

    if not train_file.exists():
        print(f"Error: Train file not found at {train_file}")
        print("Please ensure data is processed first.")
        return

    print(f"Loading dataset from: {train_file}")
    dataset = FoodDataset(train_file, mode='train')
    print(f"Dataset size: {len(dataset)}")

    # Try to load first 10 samples
    success_count = 0
    fail_count = 0

    print("\nTesting first 10 samples:")
    for i in range(min(10, len(dataset))):
        try:
            sample = dataset[i]
            print(f"Sample {i}: OK - image shape: {sample['image'].shape}, mask shape: {sample['mask'].shape}")
            success_count += 1
        except Exception as e:
            print(f"Sample {i}: FAILED - {e}")
            fail_count += 1

    print(f"\nTest complete!")
    print(f"Success: {success_count}/{success_count + fail_count}")
    print(f"Failed: {fail_count}/{success_count + fail_count}")

if __name__ == "__main__":
    test_dataset_loading()
