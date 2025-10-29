"""
Health check endpoints
"""
from fastapi import APIRouter

from ..dependencies import get_config
from ..models import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    config = get_config()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        device=config.device,
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    config = get_config()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        device=config.device,
    )
