"""
Complete nutrition analysis pipeline
Combines food detection, volume estimation, and USDA lookup
"""
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from loguru import logger

from ..models import FoodDetector, VolumeEstimator
from .usda_lookup import USDALookupService


class NutritionAnalyzer:
    """
    End-to-end nutrition analysis service
    """

    def __init__(
        self,
        detector: Optional[FoodDetector] = None,
        usda_service: Optional[USDALookupService] = None,
    ):
        """
        Initialize nutrition analyzer

        Args:
            detector: Food detector instance
            usda_service: USDA lookup service instance
        """
        self.detector = detector or FoodDetector()
        self.usda_service = usda_service or USDALookupService()
        self.volume_estimator = self.detector.volume_estimator

    def analyze_image(
        self,
        image: Union[np.ndarray, str, Path],
        food_labels: Optional[List[str]] = None,
        return_visualization: bool = False,
    ) -> Dict:
        """
        Complete nutrition analysis from image

        Args:
            image: Input image
            food_labels: Optional list of food labels for detected items
            return_visualization: Return visualization images

        Returns:
            Complete nutrition analysis
        """
        logger.info("Starting nutrition analysis")

        # Detect food items
        detection_results = self.detector.detect_and_analyze(
            image,
            return_visualizations=return_visualization
        )

        logger.info(f"Detected {detection_results['num_items']} food items")

        # Analyze nutrition for each item
        nutrition_results = []

        for i, food_item in enumerate(detection_results["food_items"]):
            # Get food label if provided
            food_label = food_labels[i] if food_labels and i < len(food_labels) else None

            # Estimate mass from volume
            volume_ml = food_item["volume_ml"]

            # Use provided label or attempt to identify
            if food_label:
                mass_g = self.volume_estimator.estimate_mass(volume_ml, food_label)
            else:
                # Use default density if no label
                mass_g = self.volume_estimator.estimate_mass(volume_ml, "unknown")
                food_label = "unknown"

            # Look up nutrition information
            nutrition_info = None
            if food_label != "unknown":
                nutrition_info = self._lookup_nutrition(food_label, mass_g)

            # Combine results
            item_result = {
                "item_id": i,
                "food_label": food_label,
                "volume_ml": volume_ml,
                "estimated_mass_g": mass_g,
                "confidence": food_item["confidence"],
                **food_item,
            }

            if nutrition_info:
                item_result["nutrition"] = nutrition_info
            else:
                item_result["nutrition"] = {
                    "message": "Nutrition data not available",
                    "food_label": food_label,
                }

            # Remove raw mask to reduce payload size
            item_result.pop("mask", None)

            nutrition_results.append(item_result)

        # Calculate totals
        totals = self._calculate_totals(nutrition_results)

        # Prepare final results
        analysis = {
            "total_items": len(nutrition_results),
            "items": nutrition_results,
            "totals": totals,
        }

        if return_visualization and "visualizations" in detection_results:
            analysis["visualizations"] = detection_results["visualizations"]

        logger.info("Nutrition analysis complete")
        return analysis

    def _lookup_nutrition(
        self,
        food_label: str,
        portion_g: float,
    ) -> Optional[Dict]:
        """
        Look up nutrition information from USDA

        Args:
            food_label: Food item label
            portion_g: Portion size in grams

        Returns:
            Nutrition information
        """
        try:
            # Find best match in USDA database
            food_match = self.usda_service.find_best_match(food_label)

            if food_match:
                # Get nutrition for portion
                nutrition = self.usda_service.get_nutrition_for_portion(
                    food_match["fdc_id"],
                    portion_g
                )
                return nutrition

            logger.warning(f"No nutrition data found for: {food_label}")
            return None

        except Exception as e:
            logger.error(f"Error looking up nutrition: {e}")
            return None

    def _calculate_totals(self, nutrition_results: List[Dict]) -> Dict:
        """
        Calculate total nutrition across all detected items

        Args:
            nutrition_results: List of nutrition results

        Returns:
            Total nutrition values
        """
        totals = {
            "total_volume_ml": 0.0,
            "total_mass_g": 0.0,
            "total_energy_kcal": 0.0,
            "total_protein_g": 0.0,
            "total_carbohydrate_g": 0.0,
            "total_fat_g": 0.0,
            "total_fiber_g": 0.0,
            "total_sugar_g": 0.0,
            "total_sodium_mg": 0.0,
            "total_calcium_mg": 0.0,
            "total_iron_mg": 0.0,
            "items_with_nutrition_data": 0,
        }

        for item in nutrition_results:
            totals["total_volume_ml"] += item.get("volume_ml", 0.0)
            totals["total_mass_g"] += item.get("estimated_mass_g", 0.0)

            nutrition = item.get("nutrition")
            if nutrition and isinstance(nutrition, dict) and "energy_kcal" in nutrition:
                totals["items_with_nutrition_data"] += 1
                totals["total_energy_kcal"] += nutrition.get("energy_kcal", 0.0) or 0.0
                totals["total_protein_g"] += nutrition.get("protein_g", 0.0) or 0.0
                totals["total_carbohydrate_g"] += nutrition.get("carbohydrate_g", 0.0) or 0.0
                totals["total_fat_g"] += nutrition.get("fat_g", 0.0) or 0.0
                totals["total_fiber_g"] += nutrition.get("fiber_g", 0.0) or 0.0
                totals["total_sugar_g"] += nutrition.get("sugar_g", 0.0) or 0.0
                totals["total_sodium_mg"] += nutrition.get("sodium_mg", 0.0) or 0.0
                totals["total_calcium_mg"] += nutrition.get("calcium_mg", 0.0) or 0.0
                totals["total_iron_mg"] += nutrition.get("iron_mg", 0.0) or 0.0

        return totals

    def analyze_with_manual_labels(
        self,
        image: Union[np.ndarray, str, Path],
        food_items: List[Dict[str, any]],
        return_visualization: bool = False,
    ) -> Dict:
        """
        Analyze with manually provided food labels and portions

        Args:
            image: Input image
            food_items: List of dicts with 'label' and optionally 'mass_g' or 'volume_ml'
            return_visualization: Return visualizations

        Returns:
            Complete nutrition analysis
        """
        # Detect food items first
        detection_results = self.detector.detect_and_analyze(
            image,
            return_visualizations=return_visualization
        )

        nutrition_results = []

        for i, (detected_item, manual_item) in enumerate(
            zip(detection_results["food_items"], food_items)
        ):
            food_label = manual_item["label"]

            # Use manual mass if provided, otherwise estimate from volume
            if "mass_g" in manual_item:
                mass_g = manual_item["mass_g"]
            elif "volume_ml" in manual_item:
                mass_g = self.volume_estimator.estimate_mass(
                    manual_item["volume_ml"],
                    food_label
                )
            else:
                # Use detected volume
                volume_ml = detected_item["volume_ml"]
                mass_g = self.volume_estimator.estimate_mass(volume_ml, food_label)

            # Look up nutrition
            nutrition_info = self._lookup_nutrition(food_label, mass_g)

            item_result = {
                "item_id": i,
                "food_label": food_label,
                "volume_ml": detected_item["volume_ml"],
                "estimated_mass_g": mass_g,
                "confidence": detected_item["confidence"],
                "bbox": detected_item["bbox"],
            }

            if nutrition_info:
                item_result["nutrition"] = nutrition_info

            nutrition_results.append(item_result)

        # Calculate totals
        totals = self._calculate_totals(nutrition_results)

        analysis = {
            "total_items": len(nutrition_results),
            "items": nutrition_results,
            "totals": totals,
        }

        if return_visualization and "visualizations" in detection_results:
            analysis["visualizations"] = detection_results["visualizations"]

        return analysis
