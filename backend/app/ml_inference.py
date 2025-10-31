"""
ML Inference Module for Backend
Communicates with ML microservice via HTTP for food detection and nutrition analysis
"""
from typing import Dict, List, Optional
import logging
from .ml_client import get_ml_client

logger = logging.getLogger("app.ml_inference")

async def analyze_food_image(
    image_bytes: bytes,
    food_labels: Optional[List[str]] = None,
    return_visualization: bool = False,
) -> Dict:
    """
    Analyze food image and return nutrition information
    Delegates to ML microservice via HTTP

    Args:
        image_bytes: Raw image bytes
        food_labels: Optional list of food labels for detected items
        return_visualization: Whether to return visualization images

    Returns:
        Dictionary with detection results and nutrition information
    """
    try:
        # Get ML client
        ml_client = get_ml_client()

        # Call ML microservice
        logger.info("Sending nutrition analysis request to ML microservice...")
        results = await ml_client.analyze_nutrition(
            image_bytes=image_bytes,
            food_labels=food_labels,
            return_visualization=return_visualization,
        )

        logger.info(f"Analysis complete: {results.get('total_items', 0)} items found")
        return results

    except Exception as e:
        logger.error(f"Error in analyze_food_image: {e}", exc_info=True)
        raise

async def detect_food_items(
    image_bytes: bytes,
    return_visualization: bool = False,
) -> Dict:
    """
    Detect food items in image without nutrition analysis
    Delegates to ML microservice via HTTP

    Args:
        image_bytes: Raw image bytes
        return_visualization: Whether to return visualization images

    Returns:
        Dictionary with detection results
    """
    try:
        # Get ML client
        ml_client = get_ml_client()

        # Call ML microservice
        logger.info("Sending detection request to ML microservice...")
        results = await ml_client.detect_food_items(
            image_bytes=image_bytes,
            return_visualization=return_visualization,
        )

        logger.info(f"Detection complete: {results.get('num_items', 0)} items found")
        return results

    except Exception as e:
        logger.error(f"Error in detect_food_items: {e}", exc_info=True)
        raise

def format_nutrition_response(ml_results: Dict) -> Dict:
    """
    Format ML nutrition results to match backend API response format

    Args:
        ml_results: Results from NutritionAnalyzer

    Returns:
        Formatted response matching backend API structure
    """
    items = ml_results.get("items", [])
    totals = ml_results.get("totals", {})

    # Build food items list
    food_items = []
    for item in items:
        nutrition = item.get("nutrition", {})

        food_item = {
            "item_name": item.get("food_label", "Unknown"),
            "estimated_mass_g": item.get("estimated_mass_g", 0),
            "volume_ml": item.get("volume_ml", 0),
            "confidence": item.get("confidence", 0),
            "bbox": item.get("bbox", [0, 0, 0, 0]),
            "nutrition": {
                "calories": nutrition.get("energy_kcal", 0),
                "protein_g": nutrition.get("protein_g", 0),
                "carb_g": nutrition.get("carbohydrate_g", 0),
                "fat_g": nutrition.get("fat_g", 0),
                "fiber_g": nutrition.get("fiber_g", 0),
                "sugar_g": nutrition.get("sugar_g", 0),
                "sodium_mg": nutrition.get("sodium_mg", 0),
                "saturated_fat_g": nutrition.get("saturated_fat_g", 0),
            }
        }
        food_items.append(food_item)

    # Build total nutrition
    total_nutrition = {
        "calories": totals.get("total_energy_kcal", 0),
        "protein_g": totals.get("total_protein_g", 0),
        "carb_g": totals.get("total_carbohydrate_g", 0),
        "fat_g": totals.get("total_fat_g", 0),
        "fiber_g": totals.get("total_fiber_g", 0),
        "sugar_g": totals.get("total_sugar_g", 0),
        "sodium_mg": totals.get("total_sodium_mg", 0),
        "saturated_fat_g": 0,  # Not tracked in totals
    }

    return {
        "num_items": ml_results.get("total_items", 0),
        "food_items": food_items,
        "total_nutrition": total_nutrition,
        "source": "ml_integrated",
    }
