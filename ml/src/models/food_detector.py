"""
Complete food detection model combining SAM2 and volume estimation
"""
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from models.sam2_segmentation import SAM2Segmentor
from models.volume_estimator import VolumeEstimator
from config import config


class FoodDetector(nn.Module):
    """
    Complete food detection pipeline:
    1. Segment food items with SAM2
    2. Estimate volume/portion size
    3. Prepare for nutrition lookup
    """

    def __init__(
        self,
        sam2_model_type: str = "vit_b",
        sam2_checkpoint: Optional[Path] = None,
        device: Optional[str] = None,
        use_lightweight_head: bool = True,
    ):
        """
        Initialize food detector

        Args:
            sam2_model_type: SAM2 model variant
            sam2_checkpoint: Path to SAM2 checkpoint
            device: Device to use
            use_lightweight_head: Use lightweight head for faster training
        """
        super().__init__()

        self.device = device or config.device

        # Initialize SAM2 segmentor
        logger.info("Initializing SAM2 segmentor")
        self.segmentor = SAM2Segmentor(
            model_type=sam2_model_type,
            checkpoint_path=sam2_checkpoint,
            device=self.device,
            use_lightweight_head=use_lightweight_head,
        )

        # Initialize volume estimator
        logger.info("Initializing volume estimator")
        self.volume_estimator = VolumeEstimator()

        logger.info("Food detector initialized")

    def detect_and_analyze(
        self,
        image: Union[np.ndarray, torch.Tensor, str, Path],
        automatic: bool = True,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        return_visualizations: bool = False,
    ) -> Dict:
        """
        Detect food items and estimate portions

        Args:
            image: Input image (path, numpy array, or tensor)
            automatic: Use automatic segmentation (vs. point prompts)
            point_coords: Optional point prompts for segmentation
            point_labels: Labels for point prompts
            return_visualizations: Return visualization images

        Returns:
            Dictionary with detection results
        """
        # Load and preprocess image
        image_np = self._load_image(image)

        # Segment food items
        if automatic:
            logger.info("Running automatic segmentation")
            masks = self.segmentor.segment_automatic(image_np)
            logger.info(f"SAM2 generated {len(masks)} initial masks")

            masks = self.segmentor.postprocess_masks(
                masks,
                min_area=5000,  # Increased from 1000 to filter small regions
                filter_food_regions=True,
                apply_nms=True,  # Enable NMS
                nms_iou_threshold=0.5,  # Remove masks with >50% overlap
                min_score=0.8,  # Keep only high-confidence detections
            )
        else:
            if point_coords is None:
                raise ValueError("Point coordinates required for prompted segmentation")

            logger.info("Running prompted segmentation")
            mask_array, scores, _ = self.segmentor.segment_with_points(
                image_np,
                point_coords,
                point_labels,
            )

            # Convert to list format
            masks = []
            for i, (mask, score) in enumerate(zip(mask_array, scores)):
                masks.append({
                    "segmentation": mask,
                    "area": mask.sum(),
                    "predicted_iou": float(score),
                })

        logger.info(f"Found {len(masks)} food regions")

        # Analyze each detected food item
        food_items = []
        for i, mask_dict in enumerate(masks):
            mask = mask_dict["segmentation"]

            # Estimate volume
            volume_info = self.volume_estimator.estimate_volume(
                mask,
                image=image_np,
            )

            # Generate human-readable item name
            item_name = self._generate_item_name(
                i,
                volume_info.get("area_cm2", 0.0),
                volume_info.get("volume_ml", 0.0)
            )

            # Combine information
            food_item = {
                "item_id": i,
                "item_name": item_name,
                "mask": mask,
                "area_pixels": int(mask_dict.get("area", mask.sum())),
                "bbox": self._get_bbox(mask),
                "predicted_iou": mask_dict.get("predicted_iou", 0.0),
                **volume_info,
            }

            food_items.append(food_item)

        # Prepare results
        results = {
            "num_items": len(food_items),
            "food_items": food_items,
            "image_shape": image_np.shape,
        }

        # Add visualizations if requested
        if return_visualizations:
            results["visualizations"] = self._create_visualizations(
                image_np,
                food_items
            )

        return results

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            Segmentation predictions (B, 1, H, W)
        """
        return self.segmentor(x)

    def _load_image(
        self,
        image: Union[np.ndarray, torch.Tensor, str, Path]
    ) -> np.ndarray:
        """Load and normalize image"""
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            if image.ndim == 4:
                image = image[0]  # Take first batch
            if image.shape[0] == 3:
                image = image.transpose(1, 2, 0)  # CHW -> HWC

        # Ensure uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        return image

    def _get_bbox(self, mask: np.ndarray) -> List[int]:
        """Get bounding box from mask"""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not rows.any() or not cols.any():
            return [0, 0, 0, 0]

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        return [int(x_min), int(y_min), int(x_max), int(y_max)]

    def _generate_item_name(self, item_id: int, area_cm2: float, volume_ml: float) -> str:
        """Generate human-readable item name based on size"""
        # Categorize by area
        if area_cm2 > 50:
            size = "Large"
        elif area_cm2 > 20:
            size = "Medium"
        elif area_cm2 > 5:
            size = "Small"
        else:
            size = "Tiny"

        return f"{size} Food Item {item_id + 1}"

    def _create_visualizations(
        self,
        image: np.ndarray,
        food_items: List[Dict]
    ) -> Dict[str, np.ndarray]:
        """Create visualization images"""
        visualizations = {}

        # Create overlay with all masks
        overlay = image.copy()
        mask_overlay = np.zeros_like(image)

        for i, item in enumerate(food_items):
            mask = item["mask"]
            # Generate distinct color for each item
            color = self._generate_color(i)

            # Apply colored mask
            mask_3d = np.stack([mask] * 3, axis=-1)
            mask_overlay = np.where(mask_3d, color, mask_overlay)

            # Draw bounding box
            bbox = item["bbox"]
            cv2.rectangle(
                overlay,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color.tolist(),
                2
            )

            # Add label with item name
            item_name = item.get("item_name", f"Item {i+1}")
            label = f"{item_name}: {item['volume_ml']:.0f}ml"
            cv2.putText(
                overlay,
                label,
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color.tolist(),
                2
            )

        # Blend mask overlay with original image
        alpha = 0.4
        result = cv2.addWeighted(image, 1 - alpha, mask_overlay, alpha, 0)
        result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)

        visualizations["segmentation_overlay"] = result
        visualizations["masks_only"] = mask_overlay
        visualizations["boxes_overlay"] = overlay

        return visualizations

    def _generate_color(self, index: int) -> np.ndarray:
        """Generate distinct color for visualization"""
        # Use golden ratio for color distribution
        golden_ratio = 0.618033988749895
        hue = (index * golden_ratio) % 1.0

        # Convert HSV to RGB
        c = np.array([hue, 0.8, 0.95])
        c_hsv = (c * np.array([179, 255, 255])).astype(np.uint8)
        rgb = cv2.cvtColor(c_hsv.reshape(1, 1, 3), cv2.COLOR_HSV2RGB)[0, 0]

        return rgb
