"""
Nutrition analysis endpoints
"""
import asyncio
import io
import json

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger
from PIL import Image

from ..dependencies import get_nutrition_analyzer

router = APIRouter(prefix="/api/v1", tags=["nutrition"])


@router.post("/analyze")
async def analyze_nutrition(
    file: UploadFile = File(..., description="Image file to analyze"),
    food_labels: str = Form(None, description="Comma-separated list of food labels (optional)"),
    return_visualization: bool = Form(False, description="Return visualization images"),
):
    """
    Complete nutrition analysis from image

    Detects food items and provides nutrition information from USDA database

    Args:
        file: Image file upload (JPEG, PNG)
        food_labels: Optional comma-separated list of food labels to guide detection
        return_visualization: Whether to return visualization images

    Returns:
        Complete nutrition analysis with detected items, nutrition facts, and totals
    """
    nutrition_analyzer = get_nutrition_analyzer()
    if not nutrition_analyzer:
        raise HTTPException(
            status_code=503,
            detail="Nutrition analyzer not initialized"
        )

    try:
        # Validate and read image
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}"
            )

        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image.convert("RGB"))

        # Parse food labels
        labels = None
        if food_labels:
            labels = [label.strip() for label in food_labels.split(",") if label.strip()]

        # Run analysis
        logger.info(f"Analyzing nutrition for image: {file.filename}")
        results = await asyncio.to_thread(
            nutrition_analyzer.analyze_image,
            image_np,
            food_labels=labels,
            return_visualization=return_visualization,
        )

        logger.info(f"Nutrition analysis complete: {results.get('num_items', 0)} items analyzed")
        return JSONResponse(content=results)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in nutrition analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/analyze-detailed")
async def analyze_nutrition_detailed(
    file: UploadFile = File(..., description="Image file to analyze"),
    food_items_json: str = Form(..., description='JSON array of food items: [{"label": "rice", "mass_g": 150}, ...]'),
    return_visualization: bool = Form(False, description="Return visualization images"),
):
    """
    Nutrition analysis with detailed food item specifications

    Allows precise specification of food items and their portions

    Args:
        file: Image file upload (JPEG, PNG)
        food_items_json: JSON string with food items and their specifications
        return_visualization: Whether to return visualization images

    Returns:
        Detailed nutrition analysis for specified items
    """
    nutrition_analyzer = get_nutrition_analyzer()
    if not nutrition_analyzer:
        raise HTTPException(
            status_code=503,
            detail="Nutrition analyzer not initialized"
        )

    try:
        # Parse food items JSON
        try:
            food_items = json.loads(food_items_json)
            if not isinstance(food_items, list):
                raise ValueError("food_items must be a JSON array")
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON in food_items_json: {e}"
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image.convert("RGB"))

        # Run analysis
        logger.info(f"Analyzing nutrition with {len(food_items)} specified items")
        results = await asyncio.to_thread(
            nutrition_analyzer.analyze_with_manual_labels,
            image_np,
            food_items=food_items,
            return_visualization=return_visualization,
        )

        return JSONResponse(content=results)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detailed nutrition analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
