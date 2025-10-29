"""
FastAPI application for food detection and nutrition analysis service

Refactored with class-based structure, routers, and async support
"""
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from ..config import config
from ..models import FoodDetector
from ..services import NutritionAnalyzer, USDALookupService
from . import dependencies
from .routers import detection_router, health_router, nutrition_router, usda_router


class FoodAnalysisAPI:
    """Main API application class"""

    def __init__(self):
        """Initialize the FastAPI application"""
        self.app = FastAPI(
            title="Food Detection & Nutrition Analysis API",
            description=(
                "Modern ML pipeline using SAM2 for food detection "
                "and USDA FoodData Central for nutrition analysis"
            ),
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )

        # Configure CORS
        self._setup_middleware()

        # Register routers
        self._register_routers()

        # Register lifecycle events
        self._register_events()

    def _setup_middleware(self):
        """Setup middleware (CORS, etc.)"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _register_routers(self):
        """Register all API routers"""
        self.app.include_router(health_router)
        self.app.include_router(detection_router)
        self.app.include_router(nutrition_router)
        self.app.include_router(usda_router)

    def _register_events(self):
        """Register startup and shutdown events"""

        @self.app.on_event("startup")
        async def startup():
            """Initialize services on startup"""
            await self.initialize_services()

        @self.app.on_event("shutdown")
        async def shutdown():
            """Cleanup on shutdown"""
            await self.cleanup_services()

    async def initialize_services(self):
        """Initialize all services (models, databases, etc.)"""
        try:
            logger.info("=" * 60)
            logger.info("Initializing Food Analysis API services...")
            logger.info("=" * 60)

            # Initialize USDA service
            logger.info("📚 Loading USDA nutrition database...")
            usda_service = USDALookupService()
            dependencies.set_usda_service(usda_service)
            logger.info("✓ USDA service initialized")

            # Initialize food detector
            logger.info(f"🤖 Loading SAM2 food detection model (device: {config.device})...")
            food_detector = FoodDetector(device=config.device)
            dependencies.set_food_detector(food_detector)
            logger.info("✓ Food detector initialized")

            # Initialize nutrition analyzer
            logger.info("🔬 Initializing nutrition analyzer...")
            nutrition_analyzer = NutritionAnalyzer(
                detector=food_detector,
                usda_service=usda_service,
            )
            dependencies.set_nutrition_analyzer(nutrition_analyzer)
            logger.info("✓ Nutrition analyzer initialized")

            logger.info("=" * 60)
            logger.info("✅ All services initialized successfully!")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"❌ Failed to initialize services: {e}", exc_info=True)
            raise

    async def cleanup_services(self):
        """Cleanup services on shutdown"""
        logger.info("Shutting down services...")
        dependencies.cleanup_services()
        logger.info("Services shut down successfully")


# Create global app instance
api = FoodAnalysisAPI()
app = api.app


def run_server(
    host: Optional[str] = None,
    port: Optional[int] = None,
    reload: bool = False,
):
    """
    Run the FastAPI server

    Args:
        host: Server host (default: from config)
        port: Server port (default: from config)
        reload: Enable auto-reload (default: from config)
    """
    host = host or config.api_host
    port = port or config.api_port
    reload = reload or config.api_reload

    logger.info(f"🚀 Starting Food Analysis API server on {host}:{port}")
    logger.info(f"📖 API documentation: http://{host}:{port}/docs")

    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=config.log_level.lower(),
    )


if __name__ == "__main__":
    run_server()
