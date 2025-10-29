#!/usr/bin/env python3
"""
Data preprocessing script
Processes all raw datasets and creates train/val/test splits
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.config import config
from src.data.preprocessing import DataPreprocessor


def main():
    """Main preprocessing function"""
    logger.info("Starting data preprocessing")
    logger.info(f"Raw data path: {config.raw_data_path}")
    logger.info(f"Output path: {config.processed_data_path}")

    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Process all datasets
    try:
        df = preprocessor.process_all_datasets()
        logger.info(f"Preprocessing completed successfully!")
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Datasets: {df['dataset'].value_counts().to_dict()}")

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    main()
