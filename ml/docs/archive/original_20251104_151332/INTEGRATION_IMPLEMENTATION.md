# Integration Implementation Guide

## Overview

Step-by-step implementation of the hybrid ML + Ollama/LLaVA system.

## Architecture Recap

```
Flutter → noon_backend → [ML Service + Ollama/LLaVA] → Combined Response
```

## Implementation: noon_backend Orchestrator

### Option 1: Python (FastAPI) - Recommended

**File: `noon_backend/main.py`**

```python
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import httpx
import base64
import json
import asyncio

app = FastAPI(title="noon_backend Orchestrator")

# CORS for Flutter web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
ML_SERVICE_URL = "http://localhost:8000"
OLLAMA_URL = "http://localhost:11434"

# Models
class AnalysisRequest(BaseModel):
    image: str  # base64 encoded
    query: Optional[str] = None
    mode: str = "hybrid"  # "fast" | "accurate" | "hybrid"

class FoodAnalyzer:
    """Orchestrator for ML + LLaVA hybrid analysis"""

    def __init__(self):
        self.ml_url = ML_SERVICE_URL
        self.ollama_url = OLLAMA_URL

    async def analyze(
        self,
        image_bytes: bytes,
        query: Optional[str] = None,
        mode: str = "hybrid"
    ):
        """Main analysis pipeline"""

        if mode == "fast":
            # Quick LLaVA-only analysis
            return await self._llava_only(image_bytes, query)

        elif mode == "accurate":
            # ML-only analysis
            return await self._ml_only(image_bytes)

        else:  # hybrid
            return await self._hybrid_analysis(image_bytes, query)

    async def _hybrid_analysis(self, image_bytes: bytes, query: Optional[str]):
        """Hybrid: ML for accuracy + LLaVA for insights"""

        # Step 1: Get accurate ML analysis
        ml_results = await self._call_ml_service(image_bytes)

        if not ml_results or "error" in ml_results:
            return {"error": "ML analysis failed", "details": ml_results}

        # Step 2: Enhance with LLaVA intelligence
        context = self._build_llava_context(ml_results, query)
        llava_insights = await self._call_llava(context, image_bytes)

        # Step 3: Combine results
        return self._combine_results(ml_results, llava_insights, query)

    async def _call_ml_service(self, image_bytes: bytes):
        """Call ML service for food detection"""

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
                response = await client.post(
                    f"{self.ml_url}/api/v1/analyze",
                    files=files
                )
                response.raise_for_status()
                return response.json()

        except Exception as e:
            return {"error": str(e)}

    async def _call_llava(self, prompt: str, image_bytes: bytes):
        """Call Ollama/LLaVA for intelligent insights"""

        try:
            # Encode image for Ollama
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": "llava",
                        "prompt": prompt,
                        "images": [image_b64],
                        "stream": False
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result.get("response", "")

        except Exception as e:
            return f"Error: {str(e)}"

    def _build_llava_context(self, ml_results: dict, user_query: Optional[str]):
        """Build structured prompt for LLaVA using ML results"""

        items = ml_results.get("food_items", [])
        total_nutrition = ml_results.get("total_nutrition", {})

        # Format detected items
        items_text = "\n".join([
            f"- {item['item_name']}: "
            f"{item.get('estimated_mass_g', 0):.0f}g, "
            f"{item.get('nutrition', {}).get('calories', 0):.0f} calories"
            for item in items
        ])

        prompt = f"""You are a nutrition expert analyzing a meal.

**Detected Foods (from accurate ML analysis):**
{items_text}

**Total Nutrition:**
- Calories: {total_nutrition.get('calories', 0):.0f} kcal
- Protein: {total_nutrition.get('protein_g', 0):.1f}g
- Carbohydrates: {total_nutrition.get('carb_g', 0):.1f}g
- Fat: {total_nutrition.get('fat_g', 0):.1f}g

**User Question:** {user_query or "Analyze this meal's nutritional balance and provide insights."}

**Instructions:**
1. Use ONLY the detected foods listed above (don't try to detect foods yourself)
2. Provide insights on nutritional balance
3. Give health recommendations
4. Suggest improvements if needed
5. Be concise and practical

**Your Analysis:**"""

        return prompt

    def _combine_results(self, ml_results: dict, llava_response: str, query: Optional[str]):
        """Combine ML accuracy with LLaVA intelligence"""

        return {
            "status": "success",
            "analysis": {
                # Accurate data from ML
                "detected_items": ml_results.get("food_items", []),
                "nutrition": ml_results.get("total_nutrition", {}),

                # Intelligence from LLaVA
                "insights": {
                    "summary": llava_response,
                    "source": "ollama_llava"
                },

                # Visual results
                "visualization": {
                    "segmentation_available": True,
                    "num_items": ml_results.get("num_items", 0)
                }
            },
            "query": query,
            "mode": "hybrid",
            "sources": {
                "detection": "ml_service",
                "insights": "ollama_llava"
            }
        }

    async def _ml_only(self, image_bytes: bytes):
        """ML-only mode (accurate but no insights)"""
        ml_results = await self._call_ml_service(image_bytes)
        return {
            "status": "success",
            "analysis": ml_results,
            "mode": "accurate",
            "sources": {"detection": "ml_service"}
        }

    async def _llava_only(self, image_bytes: bytes, query: Optional[str]):
        """LLaVA-only mode (fast but less accurate)"""
        prompt = query or "Analyze the food in this image and provide nutrition information."
        llava_response = await self._call_llava(prompt, image_bytes)
        return {
            "status": "success",
            "analysis": {
                "insights": {"summary": llava_response}
            },
            "mode": "fast",
            "sources": {"detection": "ollama_llava"}
        }


# Initialize analyzer
analyzer = FoodAnalyzer()


# Routes
@app.post("/api/analyze")
async def analyze_food(
    file: UploadFile = File(...),
    query: Optional[str] = Form(None),
    mode: str = Form("hybrid")
):
    """
    Analyze food image with hybrid ML + LLaVA system

    Args:
        file: Image file (JPEG, PNG)
        query: Optional user question
        mode: "hybrid" | "accurate" | "fast"

    Returns:
        Combined analysis results
    """

    # Read image
    image_bytes = await file.read()

    # Analyze
    result = await analyzer.analyze(image_bytes, query, mode)

    return result


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "ml_service": ML_SERVICE_URL,
            "ollama": OLLAMA_URL
        }
    }


@app.get("/")
async def root():
    return {
        "service": "noon_backend Orchestrator",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/api/analyze",
            "health": "/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
```

### Running the Orchestrator

```bash
# Install dependencies
pip install fastapi uvicorn httpx python-multipart

# Run
python noon_backend/main.py

# Server starts on http://localhost:3000
```

### Testing

```bash
# Test with curl
curl -X POST http://localhost:3000/api/analyze \
  -F "file=@meal.jpg" \
  -F "query=Is this healthy?" \
  -F "mode=hybrid"

# Expected response:
{
  "status": "success",
  "analysis": {
    "detected_items": [
      {
        "item_name": "Grilled Chicken",
        "estimated_mass_g": 150,
        "nutrition": {
          "calories": 165,
          "protein_g": 31,
          ...
        }
      }
    ],
    "nutrition": {
      "calories": 436,
      "protein_g": 38,
      ...
    },
    "insights": {
      "summary": "This is a well-balanced meal with..."
    }
  }
}
```

## Option 2: Node.js (Express)

**File: `noon_backend/server.js`**

```javascript
const express = require('express');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

const ML_SERVICE_URL = 'http://localhost:8000';
const OLLAMA_URL = 'http://localhost:11434';

class FoodAnalyzer {
  async analyze(imageBuffer, query = null, mode = 'hybrid') {
    if (mode === 'hybrid') {
      return await this.hybridAnalysis(imageBuffer, query);
    } else if (mode === 'accurate') {
      return await this.mlOnly(imageBuffer);
    } else {
      return await this.llavaOnly(imageBuffer, query);
    }
  }

  async hybridAnalysis(imageBuffer, query) {
    // Call ML service
    const mlResults = await this.callMLService(imageBuffer);

    if (mlResults.error) {
      return { error: 'ML analysis failed', details: mlResults.error };
    }

    // Call LLaVA with context
    const context = this.buildLLaVAContext(mlResults, query);
    const llavaInsights = await this.callLLaVA(context, imageBuffer);

    // Combine
    return this.combineResults(mlResults, llavaInsights, query);
  }

  async callMLService(imageBuffer) {
    try {
      const formData = new FormData();
      formData.append('file', imageBuffer, {
        filename: 'image.jpg',
        contentType: 'image/jpeg'
      });

      const response = await axios.post(
        `${ML_SERVICE_URL}/api/v1/analyze`,
        formData,
        {
          headers: formData.getHeaders(),
          timeout: 30000
        }
      );

      return response.data;
    } catch (error) {
      return { error: error.message };
    }
  }

  async callLLaVA(prompt, imageBuffer) {
    try {
      const imageB64 = imageBuffer.toString('base64');

      const response = await axios.post(
        `${OLLAMA_URL}/api/generate`,
        {
          model: 'llava',
          prompt: prompt,
          images: [imageB64],
          stream: false
        },
        { timeout: 60000 }
      );

      return response.data.response || '';
    } catch (error) {
      return `Error: ${error.message}`;
    }
  }

  buildLLaVAContext(mlResults, userQuery) {
    const items = mlResults.food_items || [];
    const totalNutrition = mlResults.total_nutrition || {};

    const itemsText = items.map(item =>
      `- ${item.item_name}: ${item.estimated_mass_g || 0}g, ` +
      `${item.nutrition?.calories || 0} calories`
    ).join('\n');

    return `You are a nutrition expert analyzing a meal.

**Detected Foods:**
${itemsText}

**Total Nutrition:**
- Calories: ${totalNutrition.calories || 0} kcal
- Protein: ${totalNutrition.protein_g || 0}g
- Carbs: ${totalNutrition.carb_g || 0}g
- Fat: ${totalNutrition.fat_g || 0}g

**User Question:** ${userQuery || 'Analyze this meal.'}

Provide insights on nutritional balance and recommendations.`;
  }

  combineResults(mlResults, llavaResponse, query) {
    return {
      status: 'success',
      analysis: {
        detected_items: mlResults.food_items || [],
        nutrition: mlResults.total_nutrition || {},
        insights: {
          summary: llavaResponse,
          source: 'ollama_llava'
        }
      },
      query: query,
      mode: 'hybrid'
    };
  }

  async mlOnly(imageBuffer) {
    const mlResults = await this.callMLService(imageBuffer);
    return {
      status: 'success',
      analysis: mlResults,
      mode: 'accurate'
    };
  }

  async llavaOnly(imageBuffer, query) {
    const prompt = query || 'Analyze the food in this image.';
    const llavaResponse = await this.callLLaVA(prompt, imageBuffer);
    return {
      status: 'success',
      analysis: {
        insights: { summary: llavaResponse }
      },
      mode: 'fast'
    };
  }
}

const analyzer = new FoodAnalyzer();

// Routes
app.post('/api/analyze', upload.single('file'), async (req, res) => {
  try {
    const imageBuffer = req.file.buffer;
    const query = req.body.query || null;
    const mode = req.body.mode || 'hybrid';

    const result = await analyzer.analyze(imageBuffer, query, mode);
    res.json(result);

  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
});

app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    services: {
      ml_service: ML_SERVICE_URL,
      ollama: OLLAMA_URL
    }
  });
});

app.listen(3000, () => {
  console.log('noon_backend running on http://localhost:3000');
});
```

## Flutter Client Integration

**File: `noon_frontend/lib/services/api_service.dart`**

```dart
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class FoodAnalysisService {
  final String baseUrl;

  FoodAnalysisService({
    this.baseUrl = 'http://localhost:3000',
  });

  Future<FoodAnalysisResult> analyzeFood({
    required File imageFile,
    String? query,
    AnalysisMode mode = AnalysisMode.hybrid,
  }) async {
    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/api/analyze'),
      );

      // Add image file
      request.files.add(
        await http.MultipartFile.fromPath('file', imageFile.path),
      );

      // Add optional query
      if (query != null && query.isNotEmpty) {
        request.fields['query'] = query;
      }

      // Add mode
      request.fields['mode'] = mode.name;

      // Send request
      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final json = jsonDecode(response.body);
        return FoodAnalysisResult.fromJson(json);
      } else {
        throw Exception('Analysis failed: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error analyzing food: $e');
    }
  }
}

enum AnalysisMode {
  fast,      // LLaVA only (quick)
  accurate,  // ML only (precise)
  hybrid,    // Both (best)
}

class FoodAnalysisResult {
  final String status;
  final List<FoodItem> detectedItems;
  final NutritionInfo nutrition;
  final String insights;
  final String mode;

  FoodAnalysisResult({
    required this.status,
    required this.detectedItems,
    required this.nutrition,
    required this.insights,
    required this.mode,
  });

  factory FoodAnalysisResult.fromJson(Map<String, dynamic> json) {
    final analysis = json['analysis'];

    return FoodAnalysisResult(
      status: json['status'],
      detectedItems: (analysis['detected_items'] as List?)
          ?.map((item) => FoodItem.fromJson(item))
          .toList() ?? [],
      nutrition: NutritionInfo.fromJson(analysis['nutrition'] ?? {}),
      insights: analysis['insights']?['summary'] ?? '',
      mode: json['mode'] ?? 'unknown',
    );
  }
}

class FoodItem {
  final String name;
  final double massGrams;
  final NutritionInfo nutrition;

  FoodItem({
    required this.name,
    required this.massGrams,
    required this.nutrition,
  });

  factory FoodItem.fromJson(Map<String, dynamic> json) {
    return FoodItem(
      name: json['item_name'] ?? 'Unknown',
      massGrams: (json['estimated_mass_g'] ?? 0).toDouble(),
      nutrition: NutritionInfo.fromJson(json['nutrition'] ?? {}),
    );
  }
}

class NutritionInfo {
  final double calories;
  final double proteinG;
  final double carbG;
  final double fatG;

  NutritionInfo({
    required this.calories,
    required this.proteinG,
    required this.carbG,
    required this.fatG,
  });

  factory NutritionInfo.fromJson(Map<String, dynamic> json) {
    return NutritionInfo(
      calories: (json['calories'] ?? 0).toDouble(),
      proteinG: (json['protein_g'] ?? 0).toDouble(),
      carbG: (json['carb_g'] ?? 0).toDouble(),
      fatG: (json['fat_g'] ?? 0).toDouble(),
    );
  }
}
```

**Usage in Flutter:**

```dart
// In your Flutter app
final service = FoodAnalysisService(
  baseUrl: 'http://your-server:3000',
);

// Analyze with user query
final result = await service.analyzeFood(
  imageFile: imageFile,
  query: 'Is this meal healthy?',
  mode: AnalysisMode.hybrid,
);

// Display results
print('Detected: ${result.detectedItems.length} items');
print('Calories: ${result.nutrition.calories}');
print('Insights: ${result.insights}');
```

## Deployment

### Docker Compose Setup

**File: `docker-compose.yml`**

```yaml
version: '3.8'

services:
  # ML Service
  ml_service:
    build: ./ml
    ports:
      - "8000:8000"
    volumes:
      - ./ml:/app
      - ml_models:/app/models
    environment:
      - DEVICE=cpu
    command: python run_api.py

  # Ollama
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
    command: serve

  # noon_backend Orchestrator
  noon_backend:
    build: ./noon_backend
    ports:
      - "3000:3000"
    depends_on:
      - ml_service
      - ollama
    environment:
      - ML_SERVICE_URL=http://ml_service:8000
      - OLLAMA_URL=http://ollama:11434

volumes:
  ml_models:
  ollama_models:
```

### Start All Services

```bash
# Start everything
docker-compose up -d

# Pull LLaVA model
docker exec -it noon2_ollama_1 ollama pull llava

# Check health
curl http://localhost:3000/health
```

## Summary

✅ **noon_backend orchestrator** combines ML accuracy with LLaVA intelligence
✅ **ML Service** provides precise food detection & nutrition
✅ **Ollama/LLaVA** adds conversational insights
✅ **Flutter client** gets rich, accurate results
✅ **Flexible modes** for different use cases

**Next**: Test the integration end-to-end!
