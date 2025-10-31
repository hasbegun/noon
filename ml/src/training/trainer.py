"""
Training logic for food detection model
"""
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from config import config
from data_process import create_data_loaders
from models import FoodDetector
from training.distributed import (all_reduce, barrier, get_rank, get_world_size,
                           is_main_process)
from training.losses import CombinedSegmentationLoss
from training.metrics import SegmentationMetrics, print_metrics


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

        # Loss function - use combined loss to prevent collapse to all zeros
        self.criterion = CombinedSegmentationLoss(
            bce_weight=0.4,
            dice_weight=0.4,
            focal_weight=0.2,
        )

        # Mixed precision training
        self.use_amp = config.mixed_precision and self.device == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # Training state
        self.current_epoch = 0
        self.total_epochs = config.epochs  # Default, can be overridden in train()
        self.best_val_loss = float("inf")
        self.best_f1 = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_f1_scores = []
        self.val_f1_scores = []

        # Metrics
        self.train_metrics = SegmentationMetrics(threshold=0.5)
        self.val_metrics = SegmentationMetrics(threshold=0.5)

        # Checkpoint directory
        self.checkpoint_dir = config.segmentation_models_path
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Distributed: {distributed}, Rank: {self.rank}/{self.world_size}")

    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch, returns (loss, metrics)"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Reset metrics
        self.train_metrics.reset()

        # Progress bar only on main process
        iterator = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.total_epochs}",
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

            # Monitor for prediction collapse (all zeros) and update metrics
            with torch.no_grad():
                pred_probs = torch.sigmoid(outputs)
                pred_mean = pred_probs.mean().item()
                target_mean = masks.unsqueeze(1).float().mean().item()

                # Warn if predictions are collapsing to zeros
                # Only warn if targets have positive values but predictions are near zero
                # (Don't warn when both are zero - that's correct behavior!)
                if pred_mean < 0.01 and target_mean > 0.01 and num_batches % 50 == 0:
                    logger.warning(
                        f"⚠️  Predictions may be collapsing! "
                        f"Batch {num_batches}: pred_mean={pred_mean:.6f}, "
                        f"target_mean={target_mean:.4f}"
                    )

                # Update metrics
                self.train_metrics.update(outputs, masks)

            # Update progress bar
            if is_main_process(self.rank):
                iterator.set_postfix({"loss": f"{loss.item():.4f}"})

        # Average loss across all processes
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        if self.distributed:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            all_reduce(loss_tensor)
            avg_loss = (loss_tensor / self.world_size).item()

        # Compute metrics
        metrics = self.train_metrics.compute()

        return avg_loss, metrics

    def validate(self) -> Tuple[float, Dict[str, float]]:
        """Validate the model, returns (loss, metrics)"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # Reset metrics
        self.val_metrics.reset()

        # Add progress bar for validation
        iterator = tqdm(
            self.val_loader,
            desc=f"Validation {self.current_epoch + 1}/{self.total_epochs}",
            disable=not is_main_process(self.rank),
            leave=False,  # Don't leave progress bar after completion
        )

        with torch.no_grad():
            for batch in iterator:
                images = batch["images"].to(self.device)
                masks = batch["masks"].to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks.unsqueeze(1).float())

                total_loss += loss.item()
                num_batches += 1

                # Monitor validation predictions and update metrics
                pred_probs = torch.sigmoid(outputs)
                pred_mean = pred_probs.mean().item()

                # Update metrics
                self.val_metrics.update(outputs, masks)

                # Update progress bar with current loss
                if is_main_process(self.rank):
                    iterator.set_postfix({"val_loss": f"{loss.item():.4f}", "pred_mean": f"{pred_mean:.4f}"})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Average across processes
        if self.distributed:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            all_reduce(loss_tensor)
            avg_loss = (loss_tensor / self.world_size).item()

        # Compute metrics
        metrics = self.val_metrics.compute()

        return avg_loss, metrics

    def train(self, epochs: Optional[int] = None) -> Dict:
        """
        Train the model

        Args:
            epochs: Total number of epochs to train (not additional epochs)

        Returns:
            Training history
        """
        epochs = epochs or config.epochs
        self.total_epochs = epochs  # Update instance variable for progress bar
        start_time = time.time()

        # Calculate starting epoch (for resume from checkpoint)
        start_epoch = self.current_epoch + 1 if self.current_epoch > 0 else 0

        if is_main_process(self.rank):
            if start_epoch > 0:
                logger.info(f"Resuming training from epoch {start_epoch + 1}/{epochs}")
                logger.info(f"Previous best val loss: {self.best_val_loss:.4f}")
            else:
                logger.info(f"Starting training for {epochs} epochs")

        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch

            # Train
            if is_main_process(self.rank):
                logger.info(f"Starting Epoch {epoch + 1}/{epochs} - Training...")

            train_loss, train_metrics = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_f1_scores.append(train_metrics['f1'])

            # Validate
            if is_main_process(self.rank):
                logger.info(f"Epoch {epoch + 1}/{epochs} - Training complete, starting validation...")

            val_loss, val_metrics = self.validate()
            self.val_losses.append(val_loss)
            self.val_f1_scores.append(val_metrics['f1'])

            # Update scheduler
            self.scheduler.step()

            # Log progress
            if is_main_process(self.rank):
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} Complete - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | "
                    f"Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}"
                )

                # Print detailed metrics
                print_metrics(train_metrics, prefix="Train")
                print_metrics(val_metrics, prefix="Val")

                # Save best model checkpoint (based on F1 score)
                if val_metrics['f1'] > self.best_f1:
                    self.best_f1 = val_metrics['f1']
                    self.save_checkpoint("best_model.pt")
                    logger.info(f"✓ Saved best model (F1: {val_metrics['f1']:.4f}, IoU: {val_metrics['iou']:.4f})")

                # Also save if val loss improved
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss

                # Always save last checkpoint (for resume)
                self.save_checkpoint("last_checkpoint.pt")

                # Save periodic numbered checkpoint
                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
                    logger.info(f"Saved checkpoint at epoch {epoch + 1}")

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
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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

        logger.info(f"✓ Checkpoint loaded successfully:")
        logger.info(f"  - Resuming from epoch {self.current_epoch + 1}")
        logger.info(f"  - Best val loss so far: {self.best_val_loss:.4f}")
        logger.info(f"  - Training history: {len(self.train_losses)} epochs")
