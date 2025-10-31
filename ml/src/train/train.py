#!/usr/bin/env python3
"""
Training script with multi-node distributed support
"""
import argparse
import sys
from pathlib import Path
import os

# Add parent directory to path to allow imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
project_root = src_dir.parent
sys.path.insert(0, str(src_dir))

import torch
from loguru import logger

from config import config
from data_process import create_data_loaders
from models import FoodDetector
from training import Trainer, cleanup_distributed, setup_distributed


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train food detection model")

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=config.processed_data_path,
        help="Path to processed data directory",
    )

    # Training arguments
    parser.add_argument("--epochs", type=int, default=config.epochs, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=config.batch_size, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.learning_rate, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=config.num_workers, help="Data loading workers")

    # Development mode
    parser.add_argument("--dev-mode", action="store_true", help="Development mode: use small data subset for quick iteration")
    parser.add_argument("--dev-samples", type=int, default=100, help="Number of samples in dev mode")

    # Model arguments
    parser.add_argument(
        "--model-type",
        type=str,
        default=config.sam2_model_type,
        choices=["vit_b", "vit_l", "vit_h"],
        help="SAM2 model type",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to checkpoint to resume")

    # Distributed training
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes for distributed training")
    parser.add_argument("--node-rank", type=int, default=0, help="Rank of this node")
    parser.add_argument("--master-addr", type=str, default="localhost", help="Master node address")
    parser.add_argument("--master-port", type=int, default=29500, help="Master node port")

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=config.device,
        choices=["cuda", "mps", "cpu"],
        help="Device to use",
    )

    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()

    # Setup logging
    logger.info("Food Detection Model Training")
    logger.info(f"Configuration: {vars(args)}")

    # Setup distributed training if needed
    distributed = args.num_nodes > 1

    if distributed:
        logger.info(f"Setting up distributed training with {args.num_nodes} nodes")

        # Setup environment
        import os
        os.environ["MASTER_ADDR"] = args.master_addr
        os.environ["MASTER_PORT"] = str(args.master_port)
        os.environ["WORLD_SIZE"] = str(args.num_nodes)
        os.environ["RANK"] = str(args.node_rank)

        # Initialize distributed
        rank, world_size = setup_distributed(backend="gloo")
        logger.info(f"Distributed setup complete: rank={rank}, world_size={world_size}")
    else:
        rank = 0
        world_size = 1

    try:
        # Create data loaders
        if args.dev_mode:
            logger.warning("=" * 60)
            logger.warning("🔧 DEVELOPMENT MODE ENABLED")
            logger.warning(f"   Using only {args.dev_samples} training samples")
            logger.warning(f"   Using only {args.dev_samples // 2} validation samples")
            logger.warning("   For full training, remove --dev-mode flag")
            logger.warning("=" * 60)

        logger.info("Creating data loaders")
        data_loaders = create_data_loaders(
            train_file=args.data_dir / "train.parquet",
            val_file=args.data_dir / "val.parquet",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            dev_mode=args.dev_mode,
            dev_samples=args.dev_samples,
        )

        if "train" not in data_loaders:
            raise ValueError("Training data not found")

        logger.info(f"Train batches: {len(data_loaders['train'])}")
        logger.info(f"Val batches: {len(data_loaders.get('val', []))}")

        # Create model
        logger.info(f"Creating model: {args.model_type}")
        model = FoodDetector(
            sam2_model_type=args.model_type,
            device=args.device,
        )

        # Create trainer
        logger.info("Initializing trainer")
        trainer = Trainer(
            model=model,
            train_loader=data_loaders["train"],
            val_loader=data_loaders.get("val"),
            device=args.device,
            distributed=distributed,
        )

        # Check for checkpoint to resume from
        checkpoint_to_load = None
        if args.checkpoint and args.checkpoint.exists():
            checkpoint_to_load = args.checkpoint
        else:
            # Auto-detect last checkpoint
            last_checkpoint = config.segmentation_models_path / "last_checkpoint.pt"
            if last_checkpoint.exists():
                if rank == 0:
                    logger.warning(f"Found existing checkpoint: {last_checkpoint}")
                    logger.warning("To resume training, use: --checkpoint models/segmentation/last_checkpoint.pt")
                    logger.warning("To start fresh, delete the checkpoint or use a new model name")

                    # Auto-resume (comment out these lines if you want manual resume only)
                    response = input("Resume from last checkpoint? [y/N]: ").strip().lower()
                    if response == 'y':
                        checkpoint_to_load = last_checkpoint

        # Load checkpoint if found
        if checkpoint_to_load:
            trainer.load_checkpoint(checkpoint_to_load)

        # Train
        if rank == 0:
            if checkpoint_to_load:
                logger.info(f"Continuing training to {args.epochs} total epochs")
            else:
                logger.info(f"Starting training for {args.epochs} epochs")

        history = trainer.train(epochs=args.epochs)

        # Log results
        if rank == 0:
            logger.info("Training completed!")
            logger.info(f"Best validation loss: {history['best_val_loss']:.4f}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    finally:
        if distributed:
            cleanup_distributed()


if __name__ == "__main__":
    main()
