"""
Configuration management for the food detection system
"""
import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Application configuration"""

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    data_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    models_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "models")

    @property
    def raw_data_path(self) -> Path:
        return self.data_root / "raw"

    @property
    def usda_data_path(self) -> Path:
        return self.data_root / "usda"

    @property
    def processed_data_path(self) -> Path:
        return self.data_root / "processed"

    @property
    def pretrained_models_path(self) -> Path:
        return self.models_root / "pretrained"

    @property
    def segmentation_models_path(self) -> Path:
        return self.models_root / "segmentation"

    @property
    def recognition_models_path(self) -> Path:
        return self.models_root / "recognition"

    # Training Configuration
    batch_size: int = Field(default=8, description="Training batch size")
    num_workers: int = Field(default=4, description="Number of data loading workers")
    epochs: int = Field(default=50, description="Number of training epochs")
    learning_rate: float = Field(default=1e-4, description="Learning rate")
    weight_decay: float = Field(default=1e-4, description="Weight decay for optimizer")

    # Multi-node training
    num_nodes: int = Field(default=1, ge=1, description="Number of nodes for distributed training")
    devices_per_node: int = Field(default=1, ge=1, description="Number of devices per node")

    # Model Configuration
    sam2_model_type: str = Field(default="hiera_b+", description="SAM2 model type: hiera_t, hiera_s, hiera_b+, hiera_l")
    sam2_checkpoint: Optional[str] = Field(default=None, description="Path to SAM2 checkpoint")
    image_size: int = Field(default=1024, description="Input image size for SAM2")

    # Data Processing
    train_split: float = Field(default=0.7, description="Training data split ratio")
    val_split: float = Field(default=0.15, description="Validation data split ratio")
    test_split: float = Field(default=0.15, description="Test data split ratio")

    # Data augmentation
    use_augmentation: bool = Field(default=True, description="Enable data augmentation")
    augmentation_prob: float = Field(default=0.5, description="Augmentation probability")

    # Volume Estimation
    reference_height_cm: float = Field(default=2.5, description="Reference plate/bowl height in cm")
    reference_diameter_cm: float = Field(default=25.0, description="Reference plate diameter in cm")

    # USDA Database
    usda_db_path: Optional[Path] = Field(default=None, description="Path to USDA SQLite database")
    usda_foundation_json: str = Field(
        default="FoodData_Central_foundation_food_json_2025-04-24.json",
        description="USDA foundation food JSON filename"
    )
    usda_branded_json: str = Field(
        default="FoodData_Central_branded_food_json_2025-04-24.json",
        description="USDA branded food JSON filename"
    )

    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_reload: bool = Field(default=False, description="Enable auto-reload for development")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_dir: Optional[Path] = Field(default=None, description="Log directory")

    # Device Configuration
    device: str = Field(default="mps", description="Device to use: cuda, mps, or cpu")
    mixed_precision: bool = Field(default=True, description="Enable mixed precision training")

    class Config:
        env_prefix = "FOOD_"
        env_file = ".env"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create necessary directories
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        self.pretrained_models_path.mkdir(parents=True, exist_ok=True)
        self.segmentation_models_path.mkdir(parents=True, exist_ok=True)
        self.recognition_models_path.mkdir(parents=True, exist_ok=True)

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set USDA DB path if not provided
        if self.usda_db_path is None:
            self.usda_db_path = self.usda_data_path / "nutrition.db"


# Global config instance
config = Config()
