# app/router.py

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from service import inference_service # Import our updated service instance
from api.analyze_image import handle_image_analysis

import logging
logger = logging.getLogger('app.router')
logging.basicConfig(level=logging.INFO)

router = APIRouter()

@router.post("/analyze-image/")
async def analyze_image_endpoint(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
):
    return await handle_image_analysis(background_tasks, image)

@router.post("/analyze-hybrid/")
async def analyze_hybrid_endpoint(
    image: UploadFile = File(...),
    query: str = Form(None),
    mode: str = Form("hybrid")
):
    """
    Hybrid analysis endpoint combining ML accuracy with LLaVA intelligence.

    Args:
        image: Image file to analyze
        query: Optional user question about the meal
        mode: Analysis mode - "ml_only", "hybrid", or "llava_only"

    Returns:
        Combined analysis with accurate detection and intelligent insights
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image.")

    image_bytes = await image.read()

    try:
        result = await inference_service.analyze_image_with_ml_hybrid(
            image_bytes=image_bytes,
            user_query=query,
            mode=mode
        )
        return {"status": "success", "analysis": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error during hybrid analysis: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error during analysis.")