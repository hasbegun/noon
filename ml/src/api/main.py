"""
FastAPI application for food detection and nutrition analysis service
"""
import io
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from PIL import Image
from pydantic import BaseModel

from ..config import config
from ..models import FoodDetector
from ..services import NutritionAnalyzer, USDALookupService

# Initialize FastAPI app
app = FastAPI(
    title="Food Detection & Nutrition Analysis API",
    description="Modern ML pipeline using SAM2 for food detection and USDA for nutrition analysis",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialized on startup)
food_detector: Optional[FoodDetector] = None
nutrition_analyzer: Optional[NutritionAnalyzer] = None
usda_service: Optional[USDALookupService] = None


# Pydantic models for request/response
class HealthResponse(BaseModel):
    status: str
    version: str
    device: str


class FoodItem(BaseModel):
    label: str
    mass_g: Optional[float] = None
    volume_ml: Optional[float] = None


class AnalysisRequest(BaseModel):
    food_items: Optional[List[FoodItem]] = None
    return_visualization: bool = False


class USDASearchRequest(BaseModel):
    query: str
    limit: int = 10
    category: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup"""
    global food_detector, nutrition_analyzer, usda_service

    try:
        logger.info("Initializing services...")

        # Initialize USDA service
        logger.info("Loading USDA nutrition database...")
        usda_service = USDALookupService()

        # Initialize food detector
        logger.info("Loading food detection model...")
        food_detector = FoodDetector(device=config.device)

        # Initialize nutrition analyzer
        logger.info("Initializing nutrition analyzer...")
        nutrition_analyzer = NutritionAnalyzer(
            detector=food_detector,
            usda_service=usda_service,
        )

        logger.info("All services initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down services...")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        device=config.device,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        device=config.device,
    )


@app.post("/api/v1/detect")
async def detect_food(
    file: UploadFile = File(...),
    return_visualization: bool = Form(False),
):
    """
    Detect food items in an image

    Args:
        file: Image file upload
        return_visualization: Return visualization images

    Returns:
        Detection results with bounding boxes and segmentation masks
    """
    if not food_detector:
        raise HTTPException(status_code=503, detail="Food detector not initialized")

    try:
        # Read and decode image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image.convert("RGB"))

        # Run detection
        results = food_detector.detect_and_analyze(
            image_np,
            return_visualizations=return_visualization,
        )

        # Convert numpy arrays to lists for JSON serialization
        for item in results["food_items"]:
            if "mask" in item:
                del item["mask"]  # Remove mask for response size

        return JSONResponse(content=results)

    except Exception as e:
        logger.error(f"Error in food detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/analyze")
async def analyze_nutrition(
    file: UploadFile = File(...),
    food_labels: Optional[str] = Form(None),
    return_visualization: bool = Form(False),
):
    """
    Complete nutrition analysis from image

    Args:
        file: Image file upload
        food_labels: Comma-separated list of food labels (optional)
        return_visualization: Return visualization images

    Returns:
        Complete nutrition analysis with detected items and totals
    """
    if not nutrition_analyzer:
        raise HTTPException(status_code=503, detail="Nutrition analyzer not initialized")

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image.convert("RGB"))

        # Parse food labels
        labels = None
        if food_labels:
            labels = [label.strip() for label in food_labels.split(",")]

        # Run analysis
        results = nutrition_analyzer.analyze_image(
            image_np,
            food_labels=labels,
            return_visualization=return_visualization,
        )

        return JSONResponse(content=results)

    except Exception as e:
        logger.error(f"Error in nutrition analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/analyze-detailed")
async def analyze_nutrition_detailed(
    file: UploadFile = File(...),
    food_items_json: str = Form(...),
    return_visualization: bool = Form(False),
):
    """
    Nutrition analysis with detailed food item specifications

    Args:
        file: Image file upload
        food_items_json: JSON string with food items [{"label": "rice", "mass_g": 150}, ...]
        return_visualization: Return visualization images

    Returns:
        Complete nutrition analysis
    """
    if not nutrition_analyzer:
        raise HTTPException(status_code=503, detail="Nutrition analyzer not initialized")

    try:
        import json

        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image.convert("RGB"))

        # Parse food items
        food_items = json.loads(food_items_json)

        # Run analysis
        results = nutrition_analyzer.analyze_with_manual_labels(
            image_np,
            food_items=food_items,
            return_visualization=return_visualization,
        )

        return JSONResponse(content=results)

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    except Exception as e:
        logger.error(f"Error in detailed nutrition analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/usda/search")
async def search_usda(request: USDASearchRequest):
    """
    Search USDA food database

    Args:
        request: Search request with query, limit, and category

    Returns:
        List of matching food items
    """
    if not usda_service:
        raise HTTPException(status_code=503, detail="USDA service not initialized")

    try:
        results = usda_service.search(
            query=request.query,
            limit=request.limit,
            category=request.category,
        )

        return JSONResponse(content={"results": results})

    except Exception as e:
        logger.error(f"Error searching USDA database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/usda/food/{fdc_id}")
async def get_usda_food(fdc_id: int, portion_g: float = 100.0):
    """
    Get nutrition information for a specific food item

    Args:
        fdc_id: FDC ID of food item
        portion_g: Portion size in grams

    Returns:
        Nutrition information for the specified portion
    """
    if not usda_service:
        raise HTTPException(status_code=503, detail="USDA service not initialized")

    try:
        if portion_g <= 0:
            raise HTTPException(status_code=400, detail="Portion size must be positive")

        nutrition = usda_service.get_nutrition_for_portion(fdc_id, portion_g)

        if not nutrition:
            raise HTTPException(status_code=404, detail="Food item not found")

        return JSONResponse(content=nutrition)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving USDA food: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def run_server(
    host: Optional[str] = None,
    port: Optional[int] = None,
    reload: bool = False,
):
    """
    Run the FastAPI server

    Args:
        host: Server host
        port: Server port
        reload: Enable auto-reload
    """
    host = host or config.api_host
    port = port or config.api_port

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload or config.api_reload,
        log_level=config.log_level.lower(),
    )


if __name__ == "__main__":
    run_server()
