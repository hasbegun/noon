# Integration Architecture: ML + Ollama/LLaVA Hybrid System

## Overview

Combine the accurate ML food detection system with Ollama/LLaVA for intelligent, conversational food analysis.

## Current Setup

```
┌─────────────────┐
│ Flutter Client  │
│  (noon_frontend)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  noon_backend   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Ollama + LLaVA  │ ❌ Not accurate for food
└─────────────────┘
```

## Problem

- **LLaVA**: General vision-language model, not specialized for food
- **Accuracy Issues**:
  - Incorrect food identification
  - No accurate segmentation
  - No precise nutrition data
  - Hallucinations

## Proposed Hybrid Architecture

### Option A: ML-First Pipeline (Recommended)

```
┌─────────────────────┐
│   Flutter Client    │
│   (noon_frontend)   │
└──────────┬──────────┘
           │ POST /analyze (image)
           ▼
┌─────────────────────────────────────────────────────┐
│           noon_backend (Orchestrator)               │
│                                                     │
│  Step 1: Accurate Detection                        │
│     │                                               │
│     ├──→ ML Service (ml/src/api)                   │
│     │    • SAM2 food segmentation                  │
│     │    • Volume estimation                       │
│     │    • USDA nutrition lookup                   │
│     │    Returns: Structured data                  │
│     │                                               │
│  Step 2: Intelligent Context                       │
│     │                                               │
│     ├──→ Ollama/LLaVA                              │
│     │    Context: ML results + user query          │
│     │    • Natural language explanation            │
│     │    • Meal suggestions                        │
│     │    • Health insights                         │
│     │    Returns: Conversational response          │
│     │                                               │
│  Step 3: Combine Results                           │
│     └──→ Rich response with:                       │
│          - Accurate food data (ML)                 │
│          - Natural language (LLaVA)                │
│          - Visualizations                          │
└─────────────────────────────────────────────────────┘
           │
           ▼
     ┌─────────────┐
     │   Response  │
     └─────────────┘
```

### Benefits

✅ **Accuracy**: ML provides precise food detection & nutrition
✅ **Intelligence**: LLaVA adds conversational, contextual insights
✅ **Flexibility**: Can use either independently or together
✅ **User Experience**: Best of both worlds

## Detailed Architecture

### Component Roles

#### 1. ML Service (ml/src/api/) - Accuracy Expert
**Responsibilities:**
- Food detection & segmentation (SAM2)
- Portion size estimation
- Nutrition calculation (USDA)
- Scientific accuracy

**Strengths:**
- Precise food identification
- Accurate portion sizes
- Reliable nutrition data
- No hallucinations

**API Endpoint:**
```
POST /api/v1/analyze
Input: Image file
Output: {
  "items": [
    {
      "name": "chicken breast",
      "confidence": 0.95,
      "portion_grams": 150,
      "nutrition": {...},
      "segmentation_mask": "base64...",
      "bbox": [x1, y1, x2, y2]
    }
  ],
  "total_nutrition": {...},
  "visualization_url": "..."
}
```

#### 2. Ollama/LLaVA - Intelligence Layer
**Responsibilities:**
- Natural language understanding
- Contextual insights
- Meal suggestions
- Health recommendations

**Strengths:**
- Conversational interface
- Context awareness
- Creative suggestions
- Multi-turn dialogue

**Use Cases:**
- "Is this meal healthy?"
- "What should I add for balanced nutrition?"
- "Meal prep suggestions for this dish"

#### 3. noon_backend - Orchestrator
**Responsibilities:**
- Request routing
- Result combination
- Caching
- Error handling

**Flow:**
```python
async def analyze_food(image):
    # Step 1: Get accurate ML analysis
    ml_results = await ml_service.analyze(image)

    # Step 2: Enhance with LLaVA intelligence
    context = build_context(ml_results)
    llava_insights = await ollama.query(context)

    # Step 3: Combine & return
    return combine_results(ml_results, llava_insights)
```

## Implementation Options

### Option 1: Sequential Pipeline (Simple)

```python
# noon_backend implementation
class FoodAnalyzer:
    def __init__(self):
        self.ml_service = MLServiceClient("http://localhost:8000")
        self.ollama = OllamaClient("http://localhost:11434")

    async def analyze(self, image_bytes, user_query=None):
        # 1. ML detection (always)
        ml_results = await self.ml_service.analyze(image_bytes)

        # 2. LLaVA enhancement (if query or detailed insights needed)
        if user_query or need_insights:
            prompt = self._build_prompt(ml_results, user_query)
            llava_response = await self.ollama.generate(
                model="llava",
                prompt=prompt,
                context=ml_results
            )

        # 3. Combine
        return {
            "detected_items": ml_results["items"],
            "nutrition": ml_results["total_nutrition"],
            "insights": llava_response,
            "visualization": ml_results["visualization_url"]
        }

    def _build_prompt(self, ml_results, user_query):
        """Build structured prompt for LLaVA"""
        return f"""
        Detected foods:
        {self._format_items(ml_results["items"])}

        Total nutrition:
        - Calories: {ml_results["total_nutrition"]["calories"]} kcal
        - Protein: {ml_results["total_nutrition"]["protein_g"]} g
        - Carbs: {ml_results["total_nutrition"]["carb_g"]} g
        - Fat: {ml_results["total_nutrition"]["fat_g"]} g

        User question: {user_query or "Analyze this meal"}

        Provide insights on:
        1. Nutritional balance
        2. Health considerations
        3. Suggestions for improvement
        """
```

### Option 2: Parallel Processing (Fast)

```python
async def analyze(self, image_bytes, user_query=None):
    # Run ML and LLaVA in parallel
    ml_task = asyncio.create_task(
        self.ml_service.analyze(image_bytes)
    )

    llava_task = asyncio.create_task(
        self.ollama.analyze_image(image_bytes, query=user_query)
    )

    ml_results, llava_initial = await asyncio.gather(ml_task, llava_task)

    # Use ML to correct LLaVA hallucinations
    corrected_response = self._merge_results(ml_results, llava_initial)

    return corrected_response
```

### Option 3: Smart Routing (Efficient)

```python
async def analyze(self, image_bytes, mode="hybrid"):
    """
    mode options:
    - "fast": LLaVA only (quick, less accurate)
    - "accurate": ML only (precise, no conversation)
    - "hybrid": Both (best experience)
    """

    if mode == "fast":
        return await self.ollama.analyze_image(image_bytes)

    elif mode == "accurate":
        return await self.ml_service.analyze(image_bytes)

    else:  # hybrid
        ml_results = await self.ml_service.analyze(image_bytes)

        # Use LLaVA only for insights, not detection
        llava_insights = await self.ollama.generate(
            prompt=self._build_prompt(ml_results),
            skip_detection=True  # Use ML detection
        )

        return self._combine(ml_results, llava_insights)
```

## Data Flow Examples

### Example 1: User Uploads Meal Photo

```
1. Client → noon_backend
   POST /analyze
   Body: { "image": "base64...", "query": "Is this healthy?" }

2. noon_backend → ml/src/api
   POST /api/v1/analyze
   Response: {
     "items": [
       {"name": "grilled chicken", "calories": 165, ...},
       {"name": "broccoli", "calories": 55, ...},
       {"name": "brown rice", "calories": 216, ...}
     ],
     "total_nutrition": {"calories": 436, "protein_g": 38, ...}
   }

3. noon_backend → Ollama/LLaVA
   Prompt: "Given this meal: grilled chicken (165 cal),
            broccoli (55 cal), brown rice (216 cal).
            Total: 436 calories, 38g protein.
            Is this healthy?"

   Response: "Yes! This is a well-balanced meal with:
             - Good protein source (chicken)
             - Vegetables (broccoli - vitamins)
             - Complex carbs (brown rice)
             Suggestion: Add some healthy fats like avocado..."

4. noon_backend → Client
   Combined Response:
   {
     "items": [...],  // From ML
     "nutrition": {...},  // From ML
     "insights": "Yes! This is a well-balanced...",  // From LLaVA
     "suggestions": ["Add avocado", ...],
     "visualization": "url..."
   }
```

### Example 2: Meal Planning Query

```
User: "I ate this for lunch, what should I have for dinner?"

1. ML analyzes lunch image
   Result: 800 calories, 30g protein, 90g carbs

2. LLaVA with context:
   "User had 800 cal lunch with moderate protein.
    Suggest dinner for balanced daily intake."

   Response: "For dinner, consider:
             - Higher protein (fish/tofu)
             - More vegetables
             - Lighter carbs
             Target: ~600-700 calories"
```

## API Specifications

### ML Service API (Already Exists)

**Endpoint:** `http://localhost:8000/api/v1/analyze`

```typescript
// Request
POST /api/v1/analyze
Content-Type: multipart/form-data

{
  "file": <image_file>,
  "food_labels": "optional,comma,separated"  // Optional hints
}

// Response
{
  "num_items": 3,
  "food_items": [
    {
      "item_id": 0,
      "item_name": "Grilled Chicken Breast",
      "confidence": 0.95,
      "bbox": [100, 150, 400, 450],
      "area_pixels": 90000,
      "volume_ml": 180,
      "estimated_mass_g": 150,
      "nutrition": {
        "calories": 165,
        "protein_g": 31,
        "carb_g": 0,
        "fat_g": 3.6
      }
    }
  ],
  "total_nutrition": {
    "calories": 436,
    "protein_g": 38,
    "carb_g": 45,
    "fat_g": 8.2
  },
  "visualization_url": "/visualizations/result_123.jpg"
}
```

### noon_backend Orchestrator API (New)

**Endpoint:** `http://localhost:3000/api/analyze`

```typescript
// Request
POST /api/analyze
Content-Type: application/json

{
  "image": "base64_encoded_image",
  "query": "Is this meal healthy?",  // Optional
  "mode": "hybrid",  // "fast" | "accurate" | "hybrid"
  "user_context": {  // Optional
    "dietary_restrictions": ["vegetarian"],
    "goals": ["lose_weight"],
    "daily_calories_target": 2000
  }
}

// Response
{
  "status": "success",
  "analysis": {
    // Accurate data from ML
    "detected_items": [...],
    "nutrition": {
      "total_calories": 436,
      "macros": {...},
      "breakdown_by_item": [...]
    },

    // Intelligence from LLaVA
    "insights": {
      "summary": "Balanced meal with good protein...",
      "health_score": 8.5,
      "recommendations": [
        "Add healthy fats for satiety",
        "Consider portion size..."
      ],
      "meal_context": {
        "meal_type": "lunch",
        "balance": "high_protein",
        "improvements": [...]
      }
    },

    // Visual results
    "visualization": {
      "segmentation_url": "...",
      "annotated_image_url": "..."
    }
  },

  // Metadata
  "processing_time_ms": 2500,
  "ml_confidence": 0.94,
  "sources": {
    "detection": "ml_service",
    "insights": "ollama_llava"
  }
}
```

## Implementation Steps

### Phase 1: Setup noon_backend Orchestrator

1. **Create noon_backend service** (if not exists)
   ```bash
   # FastAPI or Node.js/Express
   mkdir noon_backend
   cd noon_backend
   ```

2. **Install dependencies**
   ```bash
   # Python (FastAPI)
   pip install fastapi uvicorn httpx aiohttp pillow

   # Node.js (Express)
   npm install express axios form-data multer
   ```

3. **Create orchestrator service**
   - HTTP clients for ML service and Ollama
   - Request routing logic
   - Response combination

### Phase 2: ML Service Readiness

1. **Verify ML API is running**
   ```bash
   cd ml
   python run_api.py
   # Should be on http://localhost:8000
   ```

2. **Test ML endpoints**
   ```bash
   curl -X POST http://localhost:8000/api/v1/analyze \
     -F "file=@test.jpg" \
     -F "food_labels=chicken,rice,broccoli"
   ```

### Phase 3: Ollama/LLaVA Integration

1. **Ensure Ollama is running**
   ```bash
   ollama serve
   # Should be on http://localhost:11434
   ```

2. **Pull LLaVA model**
   ```bash
   ollama pull llava
   ```

3. **Create LLaVA client wrapper**
   - Image analysis
   - Context-aware prompting
   - Response parsing

### Phase 4: Flutter Client Updates

1. **Update API endpoint**
   ```dart
   // Change from direct Ollama to noon_backend
   final response = await http.post(
     Uri.parse('http://your-server:3000/api/analyze'),
     body: jsonEncode({
       'image': base64Image,
       'query': userQuery,
       'mode': 'hybrid'
     }),
   );
   ```

2. **Update UI for rich responses**
   - Display detected items with nutrition
   - Show LLaVA insights separately
   - Visualizations

### Phase 5: Testing & Optimization

1. **Integration tests**
2. **Performance optimization**
3. **Caching strategy**
4. **Error handling**

## Benefits Summary

| Feature | ML Only | LLaVA Only | Hybrid (Recommended) |
|---------|---------|------------|----------------------|
| Food identification accuracy | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Nutrition accuracy | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Conversational insights | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Natural language | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Portion estimation | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Meal planning advice | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Speed | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Overall** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## Next Steps

1. Review architecture and choose implementation option
2. Set up noon_backend orchestrator service
3. Implement ML service client
4. Implement Ollama/LLaVA client
5. Build combination logic
6. Update Flutter client
7. Test end-to-end
8. Deploy

---

**Recommended**: Start with **Option 1: Sequential Pipeline** for simplicity, then optimize with parallel processing if needed.

See `INTEGRATION_IMPLEMENTATION.md` for detailed code examples.
