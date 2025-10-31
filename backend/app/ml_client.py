"""
ML Microservice HTTP Client
Communicates with the ML inference microservice via HTTP
"""
import httpx
import logging
from typing import Dict, List, Optional
from pathlib import Path
import os

logger = logging.getLogger("app.ml_client")


class MLServiceClient:
    """HTTP client for communicating with ML microservice"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 300.0,  # 5 minutes for ML inference
    ):
        """
        Initialize ML service client

        Args:
            base_url: Base URL of ML service (e.g., http://ml-service:8001)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or os.getenv(
            "ML_SERVICE_URL", "http://localhost:8001"
        )
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
        )
        logger.info(f"ML Service Client initialized: {self.base_url}")

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

    async def health_check(self) -> Dict:
        """
        Check ML service health

        Returns:
            Health status dictionary

        Raises:
            httpx.HTTPError: If service is unavailable
        """
        try:
            response = await self.client.get("/health")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"ML service health check failed: {e}")
            raise

    async def detect_food_items(
        self,
        image_bytes: bytes,
        return_visualization: bool = False,
    ) -> Dict:
        """
        Detect food items in image

        Args:
            image_bytes: Raw image bytes
            return_visualization: Whether to return visualization images

        Returns:
            Detection results dictionary

        Raises:
            httpx.HTTPError: If request fails
        """
        try:
            files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
            data = {"return_visualization": str(return_visualization).lower()}

            response = await self.client.post(
                "/detect",
                files=files,
                data=data,
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPError as e:
            logger.error(f"Food detection request failed: {e}")
            raise

    async def analyze_nutrition(
        self,
        image_bytes: bytes,
        food_labels: Optional[List[str]] = None,
        return_visualization: bool = False,
    ) -> Dict:
        """
        Analyze food image for nutrition information

        Args:
            image_bytes: Raw image bytes
            food_labels: Optional list of food labels for detected items
            return_visualization: Whether to return visualization images

        Returns:
            Nutrition analysis results dictionary

        Raises:
            httpx.HTTPError: If request fails
        """
        try:
            files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
            data = {
                "return_visualization": str(return_visualization).lower(),
            }

            if food_labels:
                # Send food labels as form data (comma-separated)
                data["food_labels"] = ",".join(food_labels)

            response = await self.client.post(
                "/analyze",
                files=files,
                data=data,
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPError as e:
            logger.error(f"Nutrition analysis request failed: {e}")
            raise


# Global client instance
_ml_client: Optional[MLServiceClient] = None


def get_ml_client() -> MLServiceClient:
    """Get or create the ML service client singleton"""
    global _ml_client
    if _ml_client is None:
        _ml_client = MLServiceClient()
    return _ml_client


async def close_ml_client():
    """Close the ML service client"""
    global _ml_client
    if _ml_client is not None:
        await _ml_client.close()
        _ml_client = None
