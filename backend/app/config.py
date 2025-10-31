from pydantic_settings import BaseSettings
from pydantic import Field
import os

class Settings(BaseSettings):
    """Manages application settings and configurations.

    Environment variables can be set in .env file or via docker-compose.

    For local development (services on host):
        - OLLAMA_API_URL=http://localhost:11434/api/generate
        - LLAMACPP_API_URL=http://localhost:8088/v1/chat/completions
        - ML_SERVICE_URL=http://localhost:8001/api/v1/analyze

    For containerized deployment (accessing host services):
        - OLLAMA_API_URL=http://host.docker.internal:11434/api/generate
        - LLAMACPP_API_URL=http://host.docker.internal:8088/v1/chat/completions
        - ML_SERVICE_URL=http://host.docker.internal:8001/api/v1/analyze
    """

    # Service URLs
    OLLAMA_API_URL: str = Field(
        default="http://host.docker.internal:11434/api/generate",
        description="URL for Ollama API service"
    )

    LLAMACPP_API_URL: str = Field(
        default="http://host.docker.internal:8088/v1/chat/completions",
        description="URL for Llama.cpp API service"
    )

    # Note: ML inference is now integrated directly into the backend
    # No need for separate ML_SERVICE_URL

    # Bloom filter configuration
    BLOOMFILTER_SIZE: int = Field(
        default=100,
        description="Size of bloom filter for duplicate detection"
    )

    BLOOMFILTER_FPR: float = Field(
        default=0.000001,
        description="False positive rate for bloom filter"
    )

    # Application settings
    HOST: str = Field(
        default="0.0.0.0",
        description="Host to bind the API server"
    )

    PORT: int = Field(
        default=8000,
        description="Port to bind the API server"
    )

    DEBUG: bool = Field(
        default=False,
        description="Enable debug mode"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Create a single settings instance to be used across the application
settings = Settings()
