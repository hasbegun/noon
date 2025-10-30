"""
SAM2 (Segment Anything Model 2) integration for food segmentation
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from loguru import logger

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    SAM2_AVAILABLE = True
except ImportError:
    logger.warning("SAM2 not installed. Install with: pip install segment-anything-2")
    SAM2_AVAILABLE = False

from config import config

# Model type mapping: old names -> new names for backward compatibility
MODEL_TYPE_MAPPING = {
    "vit_b": "hiera_b+",
    "vit_l": "hiera_l",
    "vit_h": "hiera_l",  # Map to large as closest equivalent
    "vit_t": "hiera_t",
    "vit_s": "hiera_s",
}


class SAM2Segmentor(nn.Module):
    """Food segmentation using SAM2"""

    def __init__(
        self,
        model_type: str = "hiera_b+",
        checkpoint_path: Optional[Path] = None,
        device: Optional[str] = None,
        use_lightweight_head: bool = True,
    ):
        """
        Initialize SAM2 segmentor

        Args:
            model_type: SAM2 model type ('hiera_t', 'hiera_s', 'hiera_b+', 'hiera_l')
                       Also accepts legacy names: 'vit_b', 'vit_l', 'vit_h'
            checkpoint_path: Path to SAM2 checkpoint
            device: Device to use (cuda, mps, or cpu)
            use_lightweight_head: Use lightweight segmentation head for training (much faster)
        """
        super().__init__()

        if not SAM2_AVAILABLE:
            raise ImportError("SAM2 is not installed")

        self.device = device or config.device
        self.use_lightweight_head = use_lightweight_head

        # Map legacy model type names to new ones
        if model_type in MODEL_TYPE_MAPPING:
            logger.info(f"Mapping legacy model type '{model_type}' to '{MODEL_TYPE_MAPPING[model_type]}'")
            model_type = MODEL_TYPE_MAPPING[model_type]

        self.model_type = model_type

        # SAM2 model configuration
        self.model_cfg = f"sam2_{model_type}.yaml"
        self.checkpoint_path = checkpoint_path or self._get_default_checkpoint()

        # Build SAM2 model
        logger.info(f"Loading SAM2 model: {model_type}")
        self.sam2_model = self._build_sam2()

        # Only use SAM2ImagePredictor if not using placeholder
        if hasattr(self, 'use_placeholder') and self.use_placeholder:
            self.predictor = self._create_placeholder_predictor()
            self.mask_generator = None
            logger.info(f"Placeholder model loaded on {self.device}")
        else:
            self.predictor = SAM2ImagePredictor(self.sam2_model)
            self.mask_generator = SAM2AutomaticMaskGenerator(
                self.sam2_model,
                points_per_side=24,  # Reduced from 32 to generate fewer masks
                pred_iou_thresh=0.86,  # Increased from 0.8 for higher quality
                stability_score_thresh=0.95,  # Increased from 0.92
                crop_n_layers=0,  # Disabled cropping to avoid multi-scale duplicates
                crop_n_points_downscale_factor=2,
                min_mask_region_area=500,  # Increased from 100 to filter small regions
                box_nms_thresh=0.7,  # NMS threshold in mask generator itself
            )
            logger.info(f"SAM2 model loaded on {self.device}")

        # Add lightweight segmentation head for efficient training
        if use_lightweight_head:
            self._create_lightweight_head()
            logger.info("Lightweight segmentation head initialized for training")

    def _get_default_checkpoint(self) -> Path:
        """Get default checkpoint path"""
        checkpoint_name = f"sam2_{self.model_type}.pt"
        checkpoint_path = config.pretrained_models_path / checkpoint_name

        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found at {checkpoint_path}")
            logger.info("Download SAM2 checkpoints from: https://github.com/facebookresearch/sam2")

        return checkpoint_path

    def _build_sam2(self):
        """Build SAM2 model"""
        try:
            model = build_sam2(
                self.model_cfg,
                str(self.checkpoint_path),
                device=self.device,
            )
            self.use_placeholder = False
            return model
        except Exception as e:
            logger.error(f"Failed to build SAM2 model: {e}")
            # Fallback: create a simple placeholder model
            logger.warning("Using placeholder model for development")
            self.use_placeholder = True
            return self._create_placeholder_model()

    def _create_lightweight_head(self):
        """Create a lightweight segmentation head for fast training"""
        # Simple UNet-style encoder-decoder for efficient training
        self.lightweight_encoder = nn.Sequential(
            # Downsample path
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> H/2, W/2

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> H/4, W/4

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> H/8, W/8
        ).to(self.device)

        # Upsample path
        self.lightweight_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),  # -> H/4, W/4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 2, stride=2),  # -> H/2, W/2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 2, stride=2),  # -> H, W
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, 1),  # Final 1x1 conv for segmentation
        ).to(self.device)

    def _create_placeholder_model(self):
        """Create a placeholder model for development when SAM2 is not available"""
        class PlaceholderSAM2(nn.Module):
            def __init__(self, image_size=1024):
                super().__init__()
                # Add required attributes for SAM2 compatibility
                self.image_size = image_size
                self.image_encoder = None  # Placeholder
                self.mask_decoder = None  # Placeholder

                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(128, 64, 2, stride=2),
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 1, 2, stride=2),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                features = self.encoder(x)
                mask = self.decoder(features)
                return mask

        return PlaceholderSAM2(image_size=config.image_size)

    def _create_placeholder_predictor(self):
        """Create a placeholder predictor that mimics SAM2ImagePredictor interface"""
        class PlaceholderPredictor:
            def __init__(self, model):
                self.model = model
                self.image = None
                self.original_size = None
                self.input_size = None

            def set_image(self, image):
                """Set image for prediction"""
                self.image = image
                self.original_size = image.shape[:2]
                self.input_size = (1024, 1024)  # Default SAM size

            def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=True):
                """Placeholder prediction"""
                if self.image is None:
                    raise ValueError("Image not set. Call set_image first.")

                h, w = self.original_size
                # Create simple masks
                if multimask_output:
                    masks = np.zeros((3, h, w), dtype=bool)
                    scores = np.array([0.9, 0.8, 0.7])
                    logits = np.random.randn(3, h, w)
                else:
                    masks = np.zeros((1, h, w), dtype=bool)
                    scores = np.array([0.9])
                    logits = np.random.randn(1, h, w)

                # If points provided, create masks around them
                if point_coords is not None:
                    for i, (coord, label) in enumerate(zip(point_coords, point_labels)):
                        if label == 1:  # Foreground
                            x, y = int(coord[0]), int(coord[1])
                            # Create a circular region
                            radius = min(h, w) // 4
                            y_coords, x_coords = np.ogrid[:h, :w]
                            mask = (x_coords - x)**2 + (y_coords - y)**2 <= radius**2
                            if i < len(masks):
                                masks[i] = mask

                # If box provided, create mask in box region
                if box is not None:
                    x1, y1, x2, y2 = box.astype(int)
                    masks[0, y1:y2, x1:x2] = True

                return masks, scores, logits

            def generate(self, points_per_side=32, pred_iou_thresh=0.88, stability_score_thresh=0.95):
                """Generate automatic masks"""
                if self.image is None:
                    raise ValueError("Image not set. Call set_image first.")

                # Use simple segmentation for placeholder
                return self._simple_segmentation()

            def _simple_segmentation(self):
                """Simple segmentation using traditional CV"""
                gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                masks = []
                h, w = self.image.shape[:2]

                for i, contour in enumerate(contours[:10]):  # Limit to 10
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(mask, [contour], -1, 1, -1)

                    area = cv2.contourArea(contour)
                    if area < 100:  # Skip tiny regions
                        continue

                    bbox = cv2.boundingRect(contour)

                    masks.append({
                        "segmentation": mask.astype(bool),
                        "area": int(area),
                        "bbox": bbox,
                        "predicted_iou": 0.9,
                        "stability_score": 0.95,
                    })

                return masks

        return PlaceholderPredictor(self.sam2_model)

    def segment_automatic(
        self,
        image: Union[np.ndarray, torch.Tensor],
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Automatic segmentation with SAM2

        Args:
            image: Input image (H, W, 3) in RGB
            points_per_side: Number of points per side for grid
            pred_iou_thresh: IoU threshold for filtering
            stability_score_thresh: Stability score threshold

        Returns:
            List of segmentation masks with metadata
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        # Ensure uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Generate automatic masks
        if self.mask_generator is not None:
            # Use SAM2AutomaticMaskGenerator
            masks = self.mask_generator.generate(image)
        else:
            # Fallback for placeholder model
            masks = self._generate_placeholder_masks(image)

        return masks

    def segment_with_points(
        self,
        image: Union[np.ndarray, torch.Tensor],
        point_coords: np.ndarray,
        point_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment with point prompts

        Args:
            image: Input image (H, W, 3) in RGB
            point_coords: Point coordinates (N, 2) in (x, y) format
            point_labels: Point labels (N,) - 1 for foreground, 0 for background

        Returns:
            masks: Segmentation masks (N, H, W)
            scores: Quality scores for each mask
            logits: Raw logits
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        self.predictor.set_image(image)

        try:
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
        except AttributeError:
            # Fallback
            masks, scores, logits = self._predict_placeholder(image, point_coords)

        return masks, scores, logits

    def segment_with_boxes(
        self,
        image: Union[np.ndarray, torch.Tensor],
        boxes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment with bounding box prompts

        Args:
            image: Input image (H, W, 3) in RGB
            boxes: Bounding boxes (N, 4) in (x1, y1, x2, y2) format

        Returns:
            masks: Segmentation masks (N, H, W)
            scores: Quality scores
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        self.predictor.set_image(image)

        try:
            masks, scores, _ = self.predictor.predict(
                box=boxes,
                multimask_output=False,
            )
        except AttributeError:
            masks, scores = self._predict_boxes_placeholder(image, boxes)

        return masks, scores

    def _generate_placeholder_masks(self, image: np.ndarray) -> List[Dict]:
        """Generate placeholder masks for testing"""
        h, w = image.shape[:2]
        # Simple threshold-based segmentation
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        masks = []
        for i, contour in enumerate(contours[:10]):  # Limit to 10 masks
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 1, -1)

            masks.append({
                "segmentation": mask,
                "area": cv2.contourArea(contour),
                "bbox": cv2.boundingRect(contour),
                "predicted_iou": 0.9,
                "stability_score": 0.95,
            })

        return masks

    def _predict_placeholder(self, image: np.ndarray, points: np.ndarray):
        """Placeholder prediction"""
        h, w = image.shape[:2]
        masks = np.zeros((3, h, w), dtype=bool)
        scores = np.array([0.9, 0.8, 0.7])
        logits = np.random.randn(3, h, w)
        return masks, scores, logits

    def _predict_boxes_placeholder(self, image: np.ndarray, boxes: np.ndarray):
        """Placeholder box prediction"""
        h, w = image.shape[:2]
        n_boxes = len(boxes)
        masks = np.zeros((n_boxes, h, w), dtype=bool)
        scores = np.ones(n_boxes) * 0.9

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            masks[i, y1:y2, x1:x2] = True

        return masks, scores

    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute Intersection over Union between two masks"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0.0
        return intersection / union

    def _non_max_suppression(
        self,
        masks: List[Dict],
        iou_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to remove overlapping masks

        Args:
            masks: List of mask dictionaries sorted by score/area
            iou_threshold: IoU threshold for suppression (default 0.5)

        Returns:
            Filtered masks after NMS
        """
        if len(masks) == 0:
            return masks

        # Sort by predicted_iou score (or stability score) descending
        masks_sorted = sorted(
            masks,
            key=lambda x: x.get("predicted_iou", x.get("stability_score", 0)),
            reverse=True
        )

        keep_masks = []
        suppress = [False] * len(masks_sorted)

        for i, mask_i in enumerate(masks_sorted):
            if suppress[i]:
                continue

            keep_masks.append(mask_i)
            mask_i_seg = mask_i["segmentation"]

            # Suppress all subsequent masks that overlap significantly
            for j in range(i + 1, len(masks_sorted)):
                if suppress[j]:
                    continue

                mask_j_seg = masks_sorted[j]["segmentation"]
                iou = self._compute_iou(mask_i_seg, mask_j_seg)

                if iou > iou_threshold:
                    suppress[j] = True

        return keep_masks

    def postprocess_masks(
        self,
        masks: List[Dict],
        min_area: int = 5000,
        filter_food_regions: bool = True,
        apply_nms: bool = True,
        nms_iou_threshold: float = 0.5,
        min_score: float = 0.8,
    ) -> List[Dict]:
        """
        Postprocess segmentation masks

        Args:
            masks: List of mask dictionaries
            min_area: Minimum mask area to keep (increased from 1000 to 5000)
            filter_food_regions: Apply heuristics to filter food regions
            apply_nms: Apply Non-Maximum Suppression to remove overlaps
            nms_iou_threshold: IoU threshold for NMS (default 0.5)
            min_score: Minimum quality score to keep (default 0.8)

        Returns:
            Filtered masks
        """
        filtered_masks = []

        for mask_dict in masks:
            mask = mask_dict["segmentation"]
            area = mask_dict.get("area", mask.sum())

            # Filter by area
            if area < min_area:
                continue

            # Filter by quality score
            score = mask_dict.get("predicted_iou", mask_dict.get("stability_score", 1.0))
            if score < min_score:
                continue

            # Additional filtering for food regions
            if filter_food_regions:
                # Check mask is not at image border (likely background)
                h, w = mask.shape
                border_sum = (
                    mask[0, :].sum() + mask[-1, :].sum() +
                    mask[:, 0].sum() + mask[:, -1].sum()
                )
                border_ratio = border_sum / (2 * h + 2 * w)

                if border_ratio > 0.3:  # Too much border contact
                    continue

            filtered_masks.append(mask_dict)

        # Apply Non-Maximum Suppression to remove overlapping detections
        if apply_nms and len(filtered_masks) > 0:
            filtered_masks = self._non_max_suppression(filtered_masks, nms_iou_threshold)

        # Sort by area (largest first)
        filtered_masks.sort(key=lambda x: x.get("area", 0), reverse=True)

        return filtered_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training and validation

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            Segmentation masks (B, 1, H, W)
        """
        # Use placeholder model if available
        if hasattr(self, 'use_placeholder') and self.use_placeholder:
            return self.sam2_model(x)

        # Use lightweight head for fast training/validation (10-100x faster than full SAM2)
        # Only use full SAM2 for final inference (when explicitly needed)
        if self.use_lightweight_head:
            features = self.lightweight_encoder(x)
            masks = self.lightweight_decoder(features)
            return masks

        # Use SAM2's internal components for proper forward pass (inference only)
        B, C, H, W = x.shape
        device = x.device

        # 1. Encode the image using SAM2's image encoder
        backbone_out = self.sam2_model.forward_image(x)

        # 2. Extract image embeddings and high-res features from backbone_out
        if isinstance(backbone_out, dict):
            # Use the last feature level from backbone_fpn for embeddings
            image_embeddings = backbone_out['backbone_fpn'][-1]

            # Extract high-resolution features if the model uses them
            if self.sam2_model.use_high_res_features_in_sam and len(backbone_out['backbone_fpn']) >= 2:
                # Use the first two feature levels for high-res features
                high_res_features = [backbone_out['backbone_fpn'][0], backbone_out['backbone_fpn'][1]]
            else:
                high_res_features = None
        else:
            image_embeddings = backbone_out
            high_res_features = None

        # 3. Create dummy prompts (no point/box prompts, just automatic segmentation)
        # Use a single dummy point with label -1 (indicates no point)
        point_coords = torch.zeros(B, 1, 2, device=device)
        point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # 4. Encode prompts
        sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )

        # 5. Decode mask
        low_res_masks, iou_predictions, _, _ = self.sam2_model.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,  # Single mask output
            repeat_image=False,
            high_res_features=high_res_features,
        )

        # 6. Upsample to original size if needed
        if low_res_masks.shape[-2:] != (H, W):
            masks = torch.nn.functional.interpolate(
                low_res_masks,
                size=(H, W),
                mode='bilinear',
                align_corners=False,
            )
        else:
            masks = low_res_masks

        return masks
