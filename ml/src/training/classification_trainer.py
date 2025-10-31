"""
Trainer for food classification models
"""
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from config import config
from training.classification_losses import CombinedClassificationLoss, ClassificationWithNutritionLoss
from training.classification_metrics import (
    ClassificationMetrics,
    NutritionRegressionMetrics,
    print_classification_metrics,
    print_nutrition_metrics,
)


class ClassificationTrainer:
    """
    Trainer for food recognition models

    This trainer is specifically designed for classification tasks,
    not segmentation. It trains models to recognize food categories.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        num_classes: int,
        device: Optional[str] = None,
        save_dir: Optional[Path] = None,
        include_nutrition: bool = False,
        class_names: Optional[list] = None,
    ):
        """
        Initialize classification trainer

        Args:
            model: Model to train (FoodRecognitionModel or FoodRecognitionWithNutrition)
            train_loader: Training data loader
            val_loader: Validation data loader
            num_classes: Number of food classes
            device: Device to use
            save_dir: Directory to save checkpoints
            include_nutrition: Whether model predicts nutrition (joint task)
            class_names: List of class names for logging
        """
        self.device = device or config.device
        self.num_classes = num_classes
        self.include_nutrition = include_nutrition
        self.class_names = class_names

        # Setup model
        self.model = model.to(self.device)

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Setup scheduler - cosine annealing with warmup
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=1e-6,
        )

        # Loss function
        if include_nutrition:
            self.criterion = ClassificationWithNutritionLoss(
                num_classes=num_classes,
                classification_weight=1.0,
                nutrition_weight=0.5,
                label_smoothing=0.1,
            )
        else:
            self.criterion = CombinedClassificationLoss(
                num_classes=num_classes,
                ce_weight=0.7,
                focal_weight=0.3,
                label_smoothing=0.1,
                focal_gamma=2.0,
            )

        # Mixed precision training (only for CUDA)
        self.use_amp = config.mixed_precision and self.device == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # Metrics
        self.train_metrics = ClassificationMetrics(num_classes, average="macro")
        self.val_metrics = ClassificationMetrics(num_classes, average="macro")

        if include_nutrition:
            self.train_nutrition_metrics = NutritionRegressionMetrics()
            self.val_nutrition_metrics = NutritionRegressionMetrics()

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': [],
            'learning_rate': [],
        }

        # Save directory
        self.save_dir = save_dir or config.models_root / "recognition"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ClassificationTrainer initialized")
        logger.info(f"  Model: {model.__class__.__name__}")
        logger.info(f"  Num classes: {num_classes}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Include nutrition: {include_nutrition}")
        logger.info(f"  Save directory: {self.save_dir}")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.train_metrics.reset()
        if self.include_nutrition:
            self.train_nutrition_metrics.reset()

        total_loss = 0.0
        total_cls_loss = 0.0
        total_nutr_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")

        for batch in pbar:
            # Get data
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    if self.include_nutrition:
                        class_logits, nutrition_pred = self.model(images)
                        nutrition_target = torch.stack([
                            batch["nutrition"]["calories"],
                            batch["nutrition"]["protein_g"],
                            batch["nutrition"]["carb_g"],
                            batch["nutrition"]["fat_g"],
                            batch["nutrition"]["mass_g"],
                        ], dim=1).to(self.device)

                        loss, cls_loss, nutr_loss = self.criterion(
                            class_logits, nutrition_pred, labels, nutrition_target
                        )
                    else:
                        class_logits = self.model(images)
                        loss = self.criterion(class_logits, labels)
                        cls_loss = loss
                        nutr_loss = torch.tensor(0.0)
            else:
                if self.include_nutrition:
                    class_logits, nutrition_pred = self.model(images)
                    nutrition_target = torch.stack([
                        batch["nutrition"]["calories"],
                        batch["nutrition"]["protein_g"],
                        batch["nutrition"]["carb_g"],
                        batch["nutrition"]["fat_g"],
                        batch["nutrition"]["mass_g"],
                    ], dim=1).to(self.device)

                    loss, cls_loss, nutr_loss = self.criterion(
                        class_logits, nutrition_pred, labels, nutrition_target
                    )
                else:
                    class_logits = self.model(images)
                    loss = self.criterion(class_logits, labels)
                    cls_loss = loss
                    nutr_loss = torch.tensor(0.0)

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Update metrics
            with torch.no_grad():
                self.train_metrics.update(class_logits.detach(), labels)

                if self.include_nutrition and "nutrition" in batch:
                    nutrition_dict = {
                        'calories': nutrition_pred[:, 0].detach(),
                        'protein_g': nutrition_pred[:, 1].detach(),
                        'carb_g': nutrition_pred[:, 2].detach(),
                        'fat_g': nutrition_pred[:, 3].detach(),
                        'mass_g': nutrition_pred[:, 4].detach(),
                    }
                    target_dict = {
                        'calories': batch["nutrition"]["calories"].to(self.device),
                        'protein_g': batch["nutrition"]["protein_g"].to(self.device),
                        'carb_g': batch["nutrition"]["carb_g"].to(self.device),
                        'fat_g': batch["nutrition"]["fat_g"].to(self.device),
                        'mass_g': batch["nutrition"]["mass_g"].to(self.device),
                    }
                    self.train_nutrition_metrics.update(nutrition_dict, target_dict)

            # Accumulate loss
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            if self.include_nutrition:
                total_nutr_loss += nutr_loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'cls_loss': f"{cls_loss.item():.4f}",
            })

            # Clear MPS cache periodically to prevent OOM
            if self.device == "mps" and num_batches % 10 == 0:
                torch.mps.empty_cache()

        # Compute metrics
        metrics = self.train_metrics.compute()
        metrics['loss'] = total_loss / num_batches
        metrics['cls_loss'] = total_cls_loss / num_batches

        if self.include_nutrition:
            nutr_metrics = self.train_nutrition_metrics.compute()
            metrics.update(nutr_metrics)
            metrics['nutr_loss'] = total_nutr_loss / num_batches

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set"""
        self.model.eval()
        self.val_metrics.reset()
        if self.include_nutrition:
            self.val_nutrition_metrics.reset()

        total_loss = 0.0
        total_cls_loss = 0.0
        total_nutr_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")

        for batch in pbar:
            # Get data
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            # Forward pass
            if self.include_nutrition:
                class_logits, nutrition_pred = self.model(images)
                nutrition_target = torch.stack([
                    batch["nutrition"]["calories"],
                    batch["nutrition"]["protein_g"],
                    batch["nutrition"]["carb_g"],
                    batch["nutrition"]["fat_g"],
                    batch["nutrition"]["mass_g"],
                ], dim=1).to(self.device)

                loss, cls_loss, nutr_loss = self.criterion(
                    class_logits, nutrition_pred, labels, nutrition_target
                )
            else:
                class_logits = self.model(images)
                loss = self.criterion(class_logits, labels)
                cls_loss = loss
                nutr_loss = torch.tensor(0.0)

            # Update metrics
            self.val_metrics.update(class_logits, labels)

            if self.include_nutrition and "nutrition" in batch:
                nutrition_dict = {
                    'calories': nutrition_pred[:, 0],
                    'protein_g': nutrition_pred[:, 1],
                    'carb_g': nutrition_pred[:, 2],
                    'fat_g': nutrition_pred[:, 3],
                    'mass_g': nutrition_pred[:, 4],
                }
                target_dict = {
                    'calories': batch["nutrition"]["calories"].to(self.device),
                    'protein_g': batch["nutrition"]["protein_g"].to(self.device),
                    'carb_g': batch["nutrition"]["carb_g"].to(self.device),
                    'fat_g': batch["nutrition"]["fat_g"].to(self.device),
                    'mass_g': batch["nutrition"]["mass_g"].to(self.device),
                }
                self.val_nutrition_metrics.update(nutrition_dict, target_dict)

            # Accumulate loss
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            if self.include_nutrition:
                total_nutr_loss += nutr_loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            # Clear MPS cache periodically
            if self.device == "mps" and num_batches % 10 == 0:
                torch.mps.empty_cache()

        # Compute metrics
        metrics = self.val_metrics.compute()
        metrics['loss'] = total_loss / num_batches
        metrics['cls_loss'] = total_cls_loss / num_batches

        if self.include_nutrition:
            nutr_metrics = self.val_nutrition_metrics.compute()
            metrics.update(nutr_metrics)
            metrics['nutr_loss'] = total_nutr_loss / num_batches

        return metrics

    def train(self, epochs: int) -> Dict:
        """
        Train for multiple epochs

        Args:
            epochs: Number of epochs to train

        Returns:
            Training history
        """
        logger.info(f"Starting training for {epochs} epochs")

        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch()

            # Validate
            if self.val_loader is not None:
                val_metrics = self.validate()
            else:
                val_metrics = {}

            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log metrics
            epoch_time = time.time() - epoch_start

            logger.info(f"\nEpoch {epoch + 1}/{epochs} - Time: {epoch_time:.1f}s - LR: {current_lr:.2e}")
            print_classification_metrics(
                train_metrics,
                prefix="Train",
                top_classes=self.class_names,
                num_top_classes=5
            )

            if val_metrics:
                print_classification_metrics(
                    val_metrics,
                    prefix="Val",
                    top_classes=self.class_names,
                    num_top_classes=5
                )

                if self.include_nutrition:
                    print_nutrition_metrics(train_metrics, prefix="Train")
                    print_nutrition_metrics(val_metrics, prefix="Val")

            # Save history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['learning_rate'].append(current_lr)

            if val_metrics:
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['accuracy'])
                self.history['val_f1'].append(val_metrics['f1'])

                # Save best model
                if val_metrics['accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['accuracy']
                    self.save_checkpoint("best_accuracy.pt")
                    logger.info(f"✓ Saved best accuracy model: {self.best_val_acc:.4f}")

                if val_metrics['f1'] > self.best_val_f1:
                    self.best_val_f1 = val_metrics['f1']
                    self.save_checkpoint("best_f1.pt")
                    logger.info(f"✓ Saved best F1 model: {self.best_val_f1:.4f}")

            # Save last checkpoint
            self.save_checkpoint("last_checkpoint.pt")

        logger.info(f"\n{'='*60}")
        logger.info(f"Training complete!")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
        logger.info(f"Best validation F1: {self.best_val_f1:.4f}")
        logger.info(f"{'='*60}\n")

        return self.history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_f1': self.best_val_f1,
            'history': self.history,
            'num_classes': self.num_classes,
            'include_nutrition': self.include_nutrition,
        }

        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_f1 = checkpoint['best_val_f1']
        self.history = checkpoint['history']

        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"  Epoch: {self.current_epoch}")
        logger.info(f"  Best val accuracy: {self.best_val_acc:.4f}")
        logger.info(f"  Best val F1: {self.best_val_f1:.4f}")
