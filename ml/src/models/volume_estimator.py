"""
Volume estimation from segmentation masks
Uses depth cues and reference objects for portion size estimation
"""
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import ConvexHull

from config import config


class VolumeEstimator:
    """Estimate food volume from segmentation masks"""

    def __init__(
        self,
        reference_height_cm: Optional[float] = None,
        reference_diameter_cm: Optional[float] = None,
    ):
        """
        Initialize volume estimator

        Args:
            reference_height_cm: Reference object height (e.g., plate height)
            reference_diameter_cm: Reference object diameter (e.g., plate diameter)
        """
        self.reference_height_cm = reference_height_cm or config.reference_height_cm
        self.reference_diameter_cm = reference_diameter_cm or config.reference_diameter_cm

    def estimate_volume(
        self,
        mask: np.ndarray,
        image: Optional[np.ndarray] = None,
        depth_map: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Estimate volume from segmentation mask

        Args:
            mask: Binary segmentation mask (H, W)
            image: Original image for context (H, W, 3)
            depth_map: Optional depth map (H, W)

        Returns:
            Dictionary with volume estimates and metrics
        """
        # Calculate 2D area in pixels
        area_pixels = mask.sum()

        # Estimate scale from image
        h, w = mask.shape
        pixels_per_cm = self._estimate_scale(mask, image)

        # Convert area to cm²
        area_cm2 = area_pixels / (pixels_per_cm ** 2)

        # Estimate height/depth
        if depth_map is not None:
            avg_height_cm = self._estimate_height_from_depth(mask, depth_map, pixels_per_cm)
        else:
            # Use heuristic based on area and shape
            avg_height_cm = self._estimate_height_heuristic(mask, area_cm2)

        # Calculate volume (simplified as area × height)
        volume_cm3 = area_cm2 * avg_height_cm

        # Convert to ml (1 cm³ = 1 ml)
        volume_ml = volume_cm3

        # Calculate additional metrics
        circularity = self._calculate_circularity(mask)
        compactness = self._calculate_compactness(mask)

        return {
            "volume_ml": float(volume_ml),
            "volume_cm3": float(volume_cm3),
            "area_cm2": float(area_cm2),
            "estimated_height_cm": float(avg_height_cm),
            "circularity": float(circularity),
            "compactness": float(compactness),
            "confidence": self._calculate_confidence(mask, circularity, compactness),
        }

    def estimate_mass(
        self,
        volume_ml: float,
        food_type: str = "unknown",
        density_g_ml: Optional[float] = None,
    ) -> float:
        """
        Estimate mass from volume

        Args:
            volume_ml: Estimated volume in ml
            food_type: Type of food for density lookup
            density_g_ml: Override density (g/ml)

        Returns:
            Estimated mass in grams
        """
        if density_g_ml is None:
            # Use food-specific density estimates
            density_g_ml = self._get_food_density(food_type)

        mass_g = volume_ml * density_g_ml
        return mass_g

    def _estimate_scale(
        self,
        mask: np.ndarray,
        image: Optional[np.ndarray] = None,
    ) -> float:
        """
        Estimate pixels per cm using reference objects

        Args:
            mask: Segmentation mask
            image: Original image

        Returns:
            Pixels per cm ratio
        """
        # Simplified: assume standard plate size
        # In production, detect plate/reference object

        h, w = mask.shape

        # Estimate based on image dimensions
        # Typical camera setup: 30cm field of view at average distance
        assumed_fov_cm = 30.0
        pixels_per_cm = max(h, w) / assumed_fov_cm

        # Adjust based on mask size relative to image
        mask_area = mask.sum()
        image_area = h * w
        relative_size = mask_area / image_area

        # If mask is large, it's likely closer to camera
        if relative_size > 0.3:
            pixels_per_cm *= 1.2
        elif relative_size < 0.1:
            pixels_per_cm *= 0.8

        return pixels_per_cm

    def _estimate_height_from_depth(
        self,
        mask: np.ndarray,
        depth_map: np.ndarray,
        pixels_per_cm: float,
    ) -> float:
        """Estimate average height from depth map"""
        masked_depth = depth_map[mask > 0]

        if len(masked_depth) == 0:
            return self.reference_height_cm

        # Calculate depth variation
        depth_std = masked_depth.std()
        depth_range = masked_depth.max() - masked_depth.min()

        # Convert depth to height (simplified)
        height_cm = depth_range / pixels_per_cm

        return height_cm

    def _estimate_height_heuristic(
        self,
        mask: np.ndarray,
        area_cm2: float,
    ) -> float:
        """
        Estimate height using shape heuristics

        Args:
            mask: Binary mask
            area_cm2: Area in cm²

        Returns:
            Estimated height in cm
        """
        # Calculate shape properties
        circularity = self._calculate_circularity(mask)

        # More circular = likely round food (use sphere/cylinder model)
        # Less circular = likely flat or irregularly stacked

        if circularity > 0.8:
            # Approximately circular - assume hemisphere or cylinder
            radius_cm = np.sqrt(area_cm2 / np.pi)
            # Assume height is 40% of diameter for typical food portions
            height_cm = radius_cm * 2 * 0.4
        else:
            # Irregular shape - use conservative flat estimate
            # with slight mounding
            height_cm = 2.0 + (area_cm2 / 50.0) * 0.5

        # Clamp to reasonable range
        height_cm = np.clip(height_cm, 0.5, 15.0)

        return height_cm

    def _calculate_circularity(self, mask: np.ndarray) -> float:
        """
        Calculate circularity (4π × area / perimeter²)
        Perfect circle = 1.0
        """
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return 0.0

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            return 0.0

        circularity = 4 * np.pi * area / (perimeter ** 2)
        return min(circularity, 1.0)

    def _calculate_compactness(self, mask: np.ndarray) -> float:
        """Calculate compactness (area / bounding box area)"""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return 0.0

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h

        if bbox_area == 0:
            return 0.0

        return area / bbox_area

    def _calculate_confidence(
        self,
        mask: np.ndarray,
        circularity: float,
        compactness: float,
    ) -> float:
        """
        Calculate confidence score for volume estimate

        Args:
            mask: Segmentation mask
            circularity: Circularity metric
            compactness: Compactness metric

        Returns:
            Confidence score [0, 1]
        """
        # Factors affecting confidence:
        # 1. Mask size (larger = more reliable)
        # 2. Shape regularity (regular shapes = more reliable)
        # 3. Coverage (avoid edge cases)

        area = mask.sum()
        h, w = mask.shape
        total_area = h * w

        # Size factor
        size_factor = min(area / (total_area * 0.5), 1.0)

        # Shape factor (prefer moderate circularity and high compactness)
        shape_factor = (circularity + compactness) / 2

        # Edge factor (penalize masks touching edges)
        edge_sum = mask[0, :].sum() + mask[-1, :].sum() + mask[:, 0].sum() + mask[:, -1].sum()
        edge_ratio = edge_sum / area if area > 0 else 1.0
        edge_factor = max(1.0 - edge_ratio, 0.0)

        # Combined confidence
        confidence = (size_factor * 0.4 + shape_factor * 0.4 + edge_factor * 0.2)

        return float(np.clip(confidence, 0.0, 1.0))

    def _get_food_density(self, food_type: str) -> float:
        """
        Get estimated density for food type

        Args:
            food_type: Food category

        Returns:
            Density in g/ml
        """
        # Common food densities (g/ml)
        density_map = {
            "unknown": 0.8,
            "rice": 0.9,
            "pasta": 0.85,
            "bread": 0.3,
            "meat": 1.0,
            "chicken": 0.95,
            "fish": 0.95,
            "vegetables": 0.6,
            "fruit": 0.8,
            "salad": 0.5,
            "soup": 1.0,
            "sauce": 1.1,
            "dairy": 1.0,
            "cheese": 1.1,
            "dessert": 0.7,
        }

        # Try to match food type to category
        food_lower = food_type.lower()
        for category, density in density_map.items():
            if category in food_lower or food_lower in category:
                return density

        return density_map["unknown"]

    def estimate_multiple_items(
        self,
        masks: list,
        images: Optional[list] = None,
    ) -> list:
        """
        Estimate volumes for multiple food items

        Args:
            masks: List of segmentation masks
            images: List of corresponding images

        Returns:
            List of volume estimates
        """
        results = []

        for i, mask in enumerate(masks):
            image = images[i] if images and i < len(images) else None
            volume_info = self.estimate_volume(mask, image)
            results.append(volume_info)

        return results
