#!/usr/bin/env python3
"""
Training script with multi-node distributed support
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from loguru import logger

from src.config import config
from src.data import create_data_loaders
from src.models import FoodDetector
from src.training import Trainer, cleanup_distributed, setup_distributed


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
        logger.info("Creating data loaders")
        data_loaders = create_data_loaders(
            train_file=args.data_dir / "train.parquet",
            val_file=args.data_dir / "val.parquet",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
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

        # Load checkpoint if provided
        if args.checkpoint and args.checkpoint.exists():
            logger.info(f"Loading checkpoint: {args.checkpoint}")
            trainer.load_checkpoint(args.checkpoint)

        # Train
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
