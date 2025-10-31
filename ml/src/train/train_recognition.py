#!/usr/bin/env python3
"""
Training script for food recognition (classification) models

This script trains models to recognize/classify food items, not segment them.
It's designed to work with Nutrition5k and Food-101 datasets.

Usage:
    # Train on Nutrition5k
    python src/train/train_recognition.py --dataset nutrition5k --epochs 50

    # Train on Food-101
    python src/train/train_recognition.py --dataset food-101 --epochs 100

    # Train with nutrition prediction
    python src/train/train_recognition.py --dataset nutrition5k --with-nutrition --epochs 50

    # Development mode (quick testing)
    python src/train/train_recognition.py --dataset food-101 --dev-mode --epochs 5
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

import torch
from loguru import logger
from torch.utils.data import DataLoader

from config import config
from data_process.classification_dataset import FoodClassificationDataset, collate_fn
from data_process.food_labels import FoodLabelManager
from models import FoodRecognitionModel, FoodRecognitionWithNutrition
from training.classification_trainer import ClassificationTrainer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train food recognition model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Data arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="food-101",
        choices=["nutrition5k", "food-101", "combined"],
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=config.processed_data_path,
        help="Path to processed data directory",
    )

    # Model arguments
    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnet_b0",
        choices=["efficientnet_b0", "efficientnet_b3", "resnet50", "mobilenet_v3_small"],
        help="Backbone architecture",
    )
    parser.add_argument(
        "--with-nutrition",
        action="store_true",
        help="Train with nutrition prediction (only for nutrition5k)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout probability",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (larger than segmentation since no masks)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size (224 for B0, 300 for B3, 380 for B4)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Number of warmup epochs",
    )

    # Data augmentation arguments
    parser.add_argument(
        "--mixup",
        action="store_true",
        help="Enable mixup data augmentation",
    )
    parser.add_argument(
        "--cutmix",
        action="store_true",
        help="Enable cutmix data augmentation",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.2,
        help="Mixup interpolation strength (beta distribution parameter)",
    )
    parser.add_argument(
        "--cutmix-alpha",
        type=float,
        default=1.0,
        help="CutMix interpolation strength (beta distribution parameter)",
    )

    # Development mode
    parser.add_argument(
        "--dev-mode",
        action="store_true",
        help="Development mode: use small subset for quick iteration",
    )
    parser.add_argument(
        "--dev-samples",
        type=int,
        default=500,
        help="Number of samples in dev mode",
    )

    # Checkpoint
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint to resume training",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze backbone weights (fine-tuning classifier only)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=config.device,
        choices=["cuda", "mps", "cpu"],
        help="Device to use",
    )

    return parser.parse_args()


def create_data_loaders(args, label_manager):
    """Create train and validation data loaders"""

    train_file = args.data_dir / "train.parquet"
    val_file = args.data_dir / "val.parquet"

    if not train_file.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_file}\n"
            f"Please run preprocessing first:\n"
            f"  python src/train/preprocess_data.py"
        )

    logger.info("Creating datasets...")

    # Create datasets
    train_dataset = FoodClassificationDataset(
        data_file=train_file,
        label_manager=label_manager,
        mode="train",
        include_nutrition=args.with_nutrition,
    )

    val_dataset = FoodClassificationDataset(
        data_file=val_file,
        label_manager=label_manager,
        mode="val",
        include_nutrition=args.with_nutrition,
    )

    # Development mode: use subset
    if args.dev_mode:
        logger.warning("="*60)
        logger.warning("🔧 DEVELOPMENT MODE ENABLED")
        logger.warning(f"   Using only {args.dev_samples} training samples")
        logger.warning(f"   Using only {args.dev_samples // 5} validation samples")
        logger.warning("   For full training, remove --dev-mode flag")
        logger.warning("="*60)

        # Use subset
        train_dataset.df = train_dataset.df.head(args.dev_samples).reset_index(drop=True)
        val_dataset.df = val_dataset.df.head(args.dev_samples // 5).reset_index(drop=True)

    # Log dataset statistics
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")

    train_stats = train_dataset.get_statistics()
    logger.info(f"Train classes: {train_stats['num_classes']}")
    if args.with_nutrition and 'nutrition_samples' in train_stats:
        logger.info(f"Samples with nutrition: {train_stats['nutrition_samples']}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device in ["cuda", "mps"] else False,
        collate_fn=collate_fn,
        drop_last=True,  # Drop last incomplete batch for stable training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device in ["cuda", "mps"] else False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, train_dataset.label_manager


def main():
    """Main training function"""
    args = parse_args()

    # Setup logging
    logger.info("="*60)
    logger.info("Food Recognition Model Training")
    logger.info("="*60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Backbone: {args.backbone}")
    logger.info(f"With nutrition: {args.with_nutrition}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Image size: {args.image_size}")
    logger.info(f"Device: {args.device}")
    if args.mixup or args.cutmix:
        logger.info(f"Augmentation: mixup={args.mixup}, cutmix={args.cutmix}")
    if args.warmup_epochs > 0:
        logger.info(f"Warmup epochs: {args.warmup_epochs}")
    if args.freeze_backbone:
        logger.info("Freeze backbone: True")
    logger.info("="*60)

    # Set image size in config
    original_image_size = config.image_size
    config.image_size = args.image_size

    # Validation
    if args.with_nutrition and args.dataset != "nutrition5k":
        logger.warning(
            f"--with-nutrition flag is designed for nutrition5k dataset, "
            f"but you're using {args.dataset}. This may not work well."
        )

    # Create label manager
    logger.info(f"Setting up label manager for {args.dataset}...")
    label_manager = FoodLabelManager(args.dataset)

    logger.info(f"Number of classes: {label_manager.num_classes}")
    logger.info(f"Sample classes: {label_manager.categories[:10]}")

    # Create data loaders
    train_loader, val_loader, label_manager = create_data_loaders(args, label_manager)

    # Create model
    logger.info(f"Creating model...")

    if args.with_nutrition:
        model = FoodRecognitionWithNutrition(
            num_classes=label_manager.num_classes,
            backbone=args.backbone,
            pretrained=True,
            dropout=args.dropout,
        )
    else:
        model = FoodRecognitionModel(
            num_classes=label_manager.num_classes,
            backbone=args.backbone,
            pretrained=True,
            dropout=args.dropout,
        )

    # Optionally freeze backbone
    if args.freeze_backbone:
        model.freeze_backbone()
        logger.info("Backbone frozen - training classifier only")

    logger.info(f"Model parameters: {model.get_num_trainable_params():,}")

    # Create trainer
    logger.info("Initializing trainer...")

    # Override config learning rate if specified
    original_lr = config.learning_rate
    if args.lr != original_lr:
        config.learning_rate = args.lr

    save_dir = config.models_root / "recognition" / f"{args.dataset}_{args.backbone}"

    trainer = ClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=label_manager.num_classes,
        device=args.device,
        save_dir=save_dir,
        include_nutrition=args.with_nutrition,
        class_names=label_manager.categories,
        use_mixup=args.mixup,
        use_cutmix=args.cutmix,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        warmup_epochs=args.warmup_epochs,
        freeze_backbone=args.freeze_backbone,
    )

    # Restore original config
    config.learning_rate = original_lr
    config.image_size = original_image_size

    # Load checkpoint if specified
    if args.checkpoint and args.checkpoint.exists():
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)

    # Train
    logger.info(f"\nStarting training for {args.epochs} epochs...")
    logger.info(f"Checkpoints will be saved to: {save_dir}\n")

    try:
        history = trainer.train(epochs=args.epochs)

        # Save final model
        final_path = save_dir / "final_model.pt"
        model.save(final_path)
        logger.info(f"\n✓ Final model saved to: {final_path}")

        # Save label mapping
        label_mapping_path = save_dir / "label_mapping.json"
        label_manager.save_mapping(label_mapping_path)
        logger.info(f"✓ Label mapping saved to: {label_mapping_path}")

        logger.info("\n" + "="*60)
        logger.info("🎉 Training completed successfully!")
        logger.info(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
        logger.info(f"Best validation F1: {trainer.best_val_f1:.4f}")
        logger.info(f"Models saved in: {save_dir}")
        logger.info("="*60)

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        logger.info(f"Last checkpoint saved in: {save_dir}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
