# ML Integration into Backend

This document explains how ML inference has been integrated directly into the backend, eliminating the need for a separate ML API service.

## Overview

Previously, the backend made HTTP calls to a separate ML service running on port 8001. Now, ML inference is integrated directly into the backend, allowing it to:

- Load and run ML models directly
- Reduce network latency
- Simplify deployment (one service instead of two)
- Share resources more efficiently

## Architecture Changes

### Before
```
Backend (port 8000) --HTTP--> ML Service (port 8001)
```

### After
```
Backend (port 8000)
  └── ML Inference Module (integrated)
      └── Imports from ../ml/src
```

## Key Components

### 1. ML Inference Module (`app/ml_inference.py`)

This module provides a clean interface for the backend to use ML models:

- `get_food_detector()` - Lazy-loads the SAM2-based food detector
- `get_nutrition_analyzer()` - Lazy-loads the nutrition analyzer
- `analyze_food_image()` - Main entry point for nutrition analysis
- `detect_food_items()` - Detection-only functionality
- `format_nutrition_response()` - Formats ML results for the API

### 2. Updated Service Layer (`app/service.py`)

The `_call_ml_service()` method now:
- Calls local ML inference instead of making HTTP requests
- Uses the same interface, making the change transparent to the rest of the code
- Handles errors locally without network-related failures

### 3. Configuration Changes

- Removed `ML_SERVICE_URL` from config
- Added comment explaining ML is now integrated
- Updated `.env.example` accordingly

## Docker Integration

### Volume Mounts

The Docker container now mounts the ML directory:

```yaml
volumes:
  - ./app:/app              # Backend code (hot-reload)
  - ../ml:/ml:ro            # ML source and models (read-only)
  - ./data:/data            # Persistent data
```

### Dependencies

Added ML dependencies to `requirements.txt`:
- `torch` - PyTorch for deep learning
- `torchvision` - Computer vision utilities
- `opencv-python` - Image processing
- `transformers` - For SAM2
- `hydra-core` - Configuration management
- And more...

### Dockerfile Updates

Enhanced build process to handle ML dependencies:
- Added build tools (cmake, g++, gcc)
- Added OpenCV runtime libraries
- Multi-stage build to keep final image size reasonable

## Usage

### Starting the Backend

The backend now automatically loads ML models on startup:

```bash
# Using Docker Compose (recommended)
make compose-up-build

# Or using Docker directly
make docker-build
make docker-run-detached
```

### API Endpoints

The API remains unchanged - same endpoints, same response format:

```python
# Example: Analyze food image
POST /api/analyze
{
  "image": <file>,
  "query": "What's in this meal?"
}

# Response includes ML-powered detection + LLaVA insights
{
  "mode": "hybrid",
  "detected_items": [...],
  "nutrition": {...},
  "insights": "..."
}
```

## Benefits

1. **Simplified Deployment**
   - One service instead of two
   - No need to manage ML service separately
   - Easier to deploy and maintain

2. **Better Performance**
   - No network overhead for ML calls
   - Direct memory access to models
   - Faster response times

3. **Resource Efficiency**
   - Shared memory for models
   - Better GPU utilization (if available)
   - Lower overall resource usage

4. **Development Workflow**
   - Easier debugging (everything in one place)
   - Simpler local development setup
   - Hot-reload works for both backend and ML code

## ML Directory Separation

The ML directory (`../ml`) remains separate and focused on:
- **Training** - Model training scripts
- **Data preparation** - Dataset processing
- **Model development** - Experimentation and research
- **Evaluation** - Model testing and benchmarking

The backend only **uses** the trained models via imports. This maintains clear separation of concerns:
- ML team focuses on training in `ml/`
- Backend team uses models from `backend/`

## File Changes Summary

### New Files
- `backend/app/ml_inference.py` - ML integration module

### Modified Files
- `backend/app/service.py` - Updated to use local ML
- `backend/app/config.py` - Removed ML_SERVICE_URL
- `backend/requirements.txt` - Added ML dependencies
- `backend/Dockerfile` - Enhanced for ML packages
- `backend/docker-compose.yml` - Added ML volume mount
- `backend/Makefile` - Updated docker commands
- `backend/.env.example` - Updated configuration docs
- `backend/DOCKER_GUIDE.md` - Updated architecture docs

## Future Enhancements

Potential improvements for the future:

1. **Model Caching**
   - Cache loaded models across requests
   - Implement model versioning
   - Support multiple model variants

2. **GPU Support**
   - Add CUDA support in Docker
   - Implement GPU detection and fallback
   - Optimize for GPU inference

3. **Performance Monitoring**
   - Add inference time tracking
   - Monitor model memory usage
   - Log model performance metrics

4. **Model Updates**
   - Hot-reload models without restart
   - A/B testing different models
   - Gradual rollout of new models

## Troubleshooting

### ML Models Not Loading

If you see errors about missing ML modules:

1. Ensure the ML directory is mounted:
   ```bash
   docker exec noon-api-server ls -la /ml
   ```

2. Check Python path inside container:
   ```bash
   docker exec noon-api-server python -c "import sys; print(sys.path)"
   ```

3. Verify ML dependencies are installed:
   ```bash
   docker exec noon-api-server pip list | grep torch
   ```

### Performance Issues

If inference is slow:

1. Check available resources:
   ```bash
   docker stats noon-api-server
   ```

2. Consider increasing container memory limits in `docker-compose.yml`:
   ```yaml
   deploy:
     resources:
       limits:
         memory: 8G
   ```

3. For GPU support, add GPU configuration to docker-compose.

### Import Errors

If you see import errors from ML modules:

1. Check that ML source is properly mounted
2. Verify the ML code structure matches expected layout
3. Check logs for detailed error messages:
   ```bash
   make docker-logs
   ```

## Conclusion

The integration of ML inference into the backend simplifies the architecture while maintaining the separation of concerns. The ML directory continues to focus on training and research, while the backend efficiently uses the trained models for inference.
