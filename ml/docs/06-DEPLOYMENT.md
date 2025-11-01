# Deployment Guide

Guide for deploying trained food recognition models to production.

> 📖 **See also**: [DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md) - Original deployment guide

---

## Quick Deployment

### 1. Export Model

```python
# Export to ONNX (recommended for production)
import torch
from src.models import FoodRecognitionModel

# Load PyTorch model
model = FoodRecognitionModel.load('models/.../best_accuracy.pt')
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    'models/recognition/model.onnx',
    input_names=['image'],
    output_names=['logits'],
    dynamic_axes={'image': {0: 'batch_size'}}
)
```

### 2. Create Inference Script

```python
# inference_api.py
import onnxruntime as ort
import numpy as np
from PIL import Image

class FoodRecognizer:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image_path):
        # Preprocess
        image = Image.open(image_path).resize((224, 224))
        image_array = np.array(image).transpose(2, 0, 1)
        image_array = image_array.astype(np.float32) / 255.0

        # Normalize
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        image_array = (image_array - mean) / std

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # Inference
        logits = self.session.run(None, {self.input_name: image_array})[0]

        # Get prediction
        pred_idx = np.argmax(logits)
        confidence = np.exp(logits[0, pred_idx]) / np.exp(logits).sum()

        return pred_idx, confidence
```

### 3. Serve with FastAPI

```python
# api/main.py
from fastapi import FastAPI, File, UploadFile
from inference_api import FoodRecognizer

app = FastAPI()
recognizer = FoodRecognizer('models/recognition/model.onnx')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file
    with open('temp.jpg', 'wb') as f:
        f.write(await file.read())

    # Predict
    pred_idx, confidence = recognizer.predict('temp.jpg')

    return {
        "class_id": int(pred_idx),
        "confidence": float(confidence)
    }
```

---

## Deployment Options

### Option 1: REST API (Recommended)

**Use FastAPI or Flask**

```bash
# Install
pip install fastapi uvicorn

# Run
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Pros**:
- ✅ Easy to integrate
- ✅ Language-agnostic
- ✅ Scalable

**Cons**:
- ❌ Network latency
- ❌ Requires server

---

### Option 2: Docker Container

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/
COPY api/ ./api/

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build
docker build -t food-recognition-api .

# Run
docker run -p 8000:8000 food-recognition-api
```

---

### Option 3: Mobile Deployment (TFLite/CoreML)

**Coming soon** - Export to mobile formats

---

## Optimization

### 1. Model Quantization

```python
# Reduce model size and improve speed
import torch.quantization

model_fp32 = FoodRecognitionModel.load('model.pt')
model_fp32.eval()

# Post-training static quantization
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Save quantized model
torch.save(model_int8.state_dict(), 'model_quantized.pt')
```

**Benefits**:
- 4x smaller model size
- 2-3x faster inference
- Minimal accuracy loss (<1%)

---

### 2. TensorRT (NVIDIA GPUs)

```bash
# Convert ONNX to TensorRT
trtexec --onnx=model.onnx \
        --saveEngine=model.trt \
        --fp16  # Use FP16 precision
```

**Benefits**:
- 5-10x faster inference
- Optimized for NVIDIA GPUs

---

### 3. Batch Inference

```python
# Process multiple images at once
images_batch = [img1, img2, img3, ...]  # List of images

# Batch preprocessing
batch_array = np.stack([preprocess(img) for img in images_batch])

# Batch inference
logits = session.run(None, {input_name: batch_array})[0]

# Get predictions
predictions = np.argmax(logits, axis=1)
```

**Benefits**:
- 2-4x throughput improvement
- Better GPU utilization

---

## Production Checklist

### Before Deployment

- [ ] Test model accuracy (>90%)
- [ ] Test inference speed (<50ms)
- [ ] Test on real-world data
- [ ] Handle edge cases
- [ ] Add error handling
- [ ] Add logging
- [ ] Add monitoring
- [ ] Security review

### Performance Targets

| Metric | Target | Excellent |
|--------|--------|-----------|
| Latency | <50ms | <20ms |
| Throughput | >20 req/s | >100 req/s |
| Memory | <2GB | <500MB |
| CPU | <50% | <30% |

---

## Monitoring

### Log Key Metrics

```python
import logging

logger = logging.getLogger('food_recognition')

# Log predictions
logger.info(f'Prediction: {class_name}, Confidence: {confidence:.2f}')

# Log errors
logger.error(f'Error processing image: {error}')

# Log performance
logger.info(f'Inference time: {duration:.2f}ms')
```

### Monitor Performance

```python
import time

start_time = time.time()
prediction = model.predict(image)
duration = (time.time() - start_time) * 1000  # ms

if duration > 50:
    logger.warning(f'Slow inference: {duration:.2f}ms')
```

---

## Security

### Input Validation

```python
# Validate image format
allowed_extensions = {'.jpg', '.jpeg', '.png'}
if not file.suffix.lower() in allowed_extensions:
    raise ValueError('Invalid file type')

# Validate image size
if file.stat().st_size > 10 * 1024 * 1024:  # 10MB
    raise ValueError('File too large')

# Validate image dimensions
image = Image.open(file)
if image.size[0] > 4096 or image.size[1] > 4096:
    raise ValueError('Image dimensions too large')
```

### Rate Limiting

```python
from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("100/minute")
async def predict(request: Request, file: UploadFile):
    # ... prediction code
    pass
```

---

## Scaling

### Horizontal Scaling

```bash
# Run multiple API instances
uvicorn api.main:app --port 8000 &
uvicorn api.main:app --port 8001 &
uvicorn api.main:app --port 8002 &

# Use load balancer (nginx, HAProxy, etc.)
```

### Vertical Scaling

```python
# Use multiple workers
uvicorn api.main:app --workers 4
```

---

## Troubleshooting Deployment

### Slow Inference

1. **Use GPU**: Set `--device cuda` or `mps`
2. **Batch requests**: Process multiple images at once
3. **Optimize model**: Use quantization or TensorRT
4. **Cache results**: Cache predictions for common images

### High Memory Usage

1. **Reduce batch size**: Process fewer images at once
2. **Use smaller model**: EfficientNet-B0 instead of B3
3. **Clear cache**: Explicitly clear GPU memory

### Low Accuracy in Production

1. **Test on production data**: See [03-TESTING.md](03-TESTING.md)
2. **Add preprocessing**: Ensure same preprocessing as training
3. **Handle edge cases**: Add validation for unusual inputs

---

## Additional Resources

- **[DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md)** - Original guide
- **[03-TESTING.md](03-TESTING.md)** - Testing guide
- **[04-TROUBLESHOOTING.md](04-TROUBLESHOOTING.md)** - Troubleshooting

---

**Ready to deploy!** Start with REST API for quick deployment, then optimize as needed.
