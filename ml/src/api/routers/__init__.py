"""
API routers package
"""
from .detection import router as detection_router
from .health import router as health_router
from .nutrition import router as nutrition_router
from .usda import router as usda_router

__all__ = [
    "health_router",
    "detection_router",
    "nutrition_router",
    "usda_router",
]
