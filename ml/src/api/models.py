"""
Pydantic models for API requests and responses
"""
from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    device: str


class FoodItem(BaseModel):
    """Food item specification"""
    label: str
    mass_g: Optional[float] = None
    volume_ml: Optional[float] = None


class DetectionRequest(BaseModel):
    """Detection request parameters"""
    return_visualization: bool = Field(default=False, description="Return visualization images")


class AnalysisRequest(BaseModel):
    """Nutrition analysis request"""
    food_items: Optional[List[FoodItem]] = None
    return_visualization: bool = False


class USDASearchRequest(BaseModel):
    """USDA search request"""
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    category: Optional[str] = Field(default=None, description="Filter by food category")


class FoodDetectionResult(BaseModel):
    """Single food detection result"""
    item_id: int
    item_name: str
    area_pixels: int
    bbox: List[int]
    predicted_iou: float
    volume_ml: float
    area_cm2: float
    confidence: float


class DetectionResponse(BaseModel):
    """Detection response"""
    num_items: int
    image_shape: List[int]
    food_items: List[FoodDetectionResult]
