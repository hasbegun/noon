"""
Dependency injection for API services

Provides global service instances that are initialized on startup
"""
from typing import Optional

from ..config import config
from ..models import FoodDetector
from ..services import NutritionAnalyzer, USDALookupService

# Global service instances
_food_detector: Optional[FoodDetector] = None
_nutrition_analyzer: Optional[NutritionAnalyzer] = None
_usda_service: Optional[USDALookupService] = None


def get_food_detector() -> Optional[FoodDetector]:
    """Get the global food detector instance"""
    return _food_detector


def get_nutrition_analyzer() -> Optional[NutritionAnalyzer]:
    """Get the global nutrition analyzer instance"""
    return _nutrition_analyzer


def get_usda_service() -> Optional[USDALookupService]:
    """Get the global USDA service instance"""
    return _usda_service


def get_config():
    """Get the global config instance"""
    return config


def set_food_detector(detector: FoodDetector):
    """Set the global food detector instance"""
    global _food_detector
    _food_detector = detector


def set_nutrition_analyzer(analyzer: NutritionAnalyzer):
    """Set the global nutrition analyzer instance"""
    global _nutrition_analyzer
    _nutrition_analyzer = analyzer


def set_usda_service(service: USDALookupService):
    """Set the global USDA service instance"""
    global _usda_service
    _usda_service = service


def cleanup_services():
    """Cleanup all global service instances"""
    global _food_detector, _nutrition_analyzer, _usda_service
    _food_detector = None
    _nutrition_analyzer = None
    _usda_service = None
