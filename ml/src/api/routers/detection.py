"""
Food detection endpoints
"""
import io

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger
from PIL import Image

from ..dependencies import get_food_detector

router = APIRouter(prefix="/api/v1", tags=["detection"])


@router.post("/detect")
async def detect_food(
    file: UploadFile = File(..., description="Image file to analyze"),
    return_visualization: bool = Form(False, description="Return visualization images"),
):
    """
    Detect food items in an image using SAM2 segmentation

    Args:
        file: Image file upload (JPEG, PNG)
        return_visualization: Whether to return visualization images

    Returns:
        Detection results with bounding boxes, volumes, and areas
    """
    food_detector = get_food_detector()
    if not food_detector:
        raise HTTPException(
            status_code=503,
            detail="Food detector not initialized. Please wait for server startup to complete."
        )

    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Expected image file."
            )

        # Read and decode image
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents))
            image_np = np.array(image.convert("RGB"))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to decode image: {str(e)}"
            )

        # Run detection
        logger.info(f"Processing image: {file.filename}, shape: {image_np.shape}")
        results = food_detector.detect_and_analyze(
            image_np,
            return_visualizations=return_visualization,
        )

        # Remove mask from response (too large for JSON)
        for item in results["food_items"]:
            if "mask" in item:
                del item["mask"]

        logger.info(f"Detection complete: found {results['num_items']} items")
        return JSONResponse(content=results)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in food detection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
