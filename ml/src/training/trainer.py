"""
Training logic for food detection model
"""
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from ..config import config
from ..data import create_data_loaders
from ..models import FoodDetector
from .distributed import (all_reduce, barrier, get_rank, get_world_size,
                           is_main_process)


class Trainer:
    """Trainer for food detection model with multi-node support"""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: Optional[str] = None,
        distributed: bool = False,
    ):
        """
        Initialize trainer

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use
            distributed: Use distributed training
        """
        self.device = device or config.device
        self.distributed = distributed
        self.rank = get_rank() if distributed else 0
        self.world_size = get_world_size() if distributed else 1

        # Setup model
        self.model = model.to(self.device)

        if distributed:
            logger.info(f"Wrapping model with DDP (rank {self.rank})")
            self.model = DDP(
                self.model,
                device_ids=None,  # MPS doesn't use device_ids
                find_unused_parameters=True,
            )

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Setup scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
        )

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Mixed precision training
        self.use_amp = config.mixed_precision and self.device == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []

        # Checkpoint directory
        self.checkpoint_dir = config.segmentation_models_path
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Distributed: {distributed}, Rank: {self.rank}/{self.world_size}")

    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Progress bar only on main process
        iterator = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{config.epochs}",
            disable=not is_main_process(self.rank),
        )

        for batch in iterator:
            images = batch["images"].to(self.device)
            masks = batch["masks"].to(self.device)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks.unsqueeze(1).float())
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks.unsqueeze(1).float())

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            if is_main_process(self.rank):
                iterator.set_postfix({"loss": f"{loss.item():.4f}"})

        # Average loss across all processes
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        if self.distributed:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            all_reduce(loss_tensor)
            avg_loss = (loss_tensor / self.world_size).item()

        return avg_loss

    def validate(self) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["images"].to(self.device)
                masks = batch["masks"].to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks.unsqueeze(1).float())

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Average across processes
        if self.distributed:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            all_reduce(loss_tensor)
            avg_loss = (loss_tensor / self.world_size).item()

        return avg_loss

    def train(self, epochs: Optional[int] = None) -> Dict:
        """
        Train the model

        Args:
            epochs: Number of epochs to train

        Returns:
            Training history
        """
        epochs = epochs or config.epochs
        start_time = time.time()

        if is_main_process(self.rank):
            logger.info(f"Starting training for {epochs} epochs")

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            # Update scheduler
            self.scheduler.step()

            # Log progress
            if is_main_process(self.rank):
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

                # Save checkpoint
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best_model.pt")
                    logger.info(f"Saved best model (val_loss: {val_loss:.4f})")

                # Save periodic checkpoint
                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

            # Synchronize processes
            if self.distributed:
                barrier()

        # Final checkpoint
        if is_main_process(self.rank):
            self.save_checkpoint("final_model.pt")

            elapsed_time = time.time() - start_time
            logger.info(f"Training completed in {elapsed_time / 60:.2f} minutes")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
        }

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename

        # Get model state dict (unwrap DDP if needed)
        model_state = (
            self.model.module.state_dict()
            if isinstance(self.model, DDP)
            else self.model.state_dict()
        )

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "config": {
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "image_size": config.image_size,
            },
        }

        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint"""
        logger.info(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer and scheduler
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load training state
        self.current_epoch = checkpoint["epoch"]
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(f"Checkpoint loaded (epoch {self.current_epoch})")
