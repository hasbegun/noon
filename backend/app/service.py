import base64
import json
import logging
from fastapi import HTTPException

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from sqlalchemy.orm import Session

from config import settings
from prompts import security_prompt, food_analysis_prompt, food_identification_prompt
from storage_service import DatabaseHandler
from ml_inference import analyze_food_image, format_nutrition_response

logger = logging.getLogger("app.service")

class InferenceService:
    """A service class to interact with different inference backends."""

    # possible models: llava, granite3.2-vision, gemma3
    def __init__(self, model: str = "gemma3"):
        """Initializes the service with a default model. llava is a versatile model supporting image and text.
        """
        self.model = model

        self.ollama_llm = ChatOllama(model=self.model,
            base_url=settings.OLLAMA_API_URL.replace("/api/generate", ""))

        self.llamacpp_llm = ChatOpenAI(
            model=self.model,
            openai_api_base=settings.LLAMACPP_API_URL.replace("/chat/completions", ""),
            openai_api_key="sk-no-key-required" # Llama.cpp server doesn't need a key
        )

    async def _call_ml_service(self, image_bytes: bytes, food_labels: list[str] = None) -> dict:
        """Use integrated ML models for accurate food detection and nutrition analysis."""
        try:
            logger.info("Running integrated ML inference...")

            # Call local ML inference
            ml_results = await analyze_food_image(
                image_bytes,
                food_labels=food_labels,
                return_visualization=False,
            )

            # Format results to match expected structure
            formatted_results = format_nutrition_response(ml_results)

            logger.info(f"ML inference complete: {formatted_results.get('num_items', 0)} items detected")
            return formatted_results

        except Exception as e:
            logger.error(f"Error in ML inference: {e}", exc_info=True)
            raise HTTPException(status_code=503, detail=f"ML inference error: {str(e)}")

    async def _generate_llava_insights(self, image_bytes: bytes, ml_results: dict, user_query: str = None) -> str:
        """Generate intelligent insights using LLaVA based on ML detection results."""
        # Build context from ML results
        food_items = ml_results.get("food_items", [])
        total_nutrition = ml_results.get("total_nutrition", {})

        items_text = "\n".join([
            f"- {item.get('item_name', 'Unknown')}: {item.get('estimated_mass_g', 0)}g "
            f"({item.get('nutrition', {}).get('calories', 0)} cal)"
            for item in food_items
        ])

        context_prompt = f"""The ML system has accurately detected the following food items:

{items_text}

Total Nutrition:
- Calories: {total_nutrition.get('calories', 0)} kcal
- Protein: {total_nutrition.get('protein_g', 0)}g
- Carbs: {total_nutrition.get('carb_g', 0)}g
- Fat: {total_nutrition.get('fat_g', 0)}g

User Question: {user_query or "Provide insights on this meal's nutritional value and health aspects."}

Provide intelligent insights about this meal, including:
1. Nutritional balance assessment
2. Health considerations
3. Suggestions for improvement or pairing
"""

        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{encoded_image}"

        message = HumanMessage(content=[
            {"type": "text", "text": context_prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ])

        try:
            response = await self.ollama_llm.ainvoke([message])
            return response.content
        except Exception as e:
            logger.error(f"Error generating LLaVA insights: {e}")
            return "Unable to generate insights at this time."

    async def _identify_food_items(self, image_bytes: bytes) -> list[dict]:
        """Step 1: Use the LLM to identify food items and quantities."""
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{encoded_image}"
        prompt = f'{food_identification_prompt} {security_prompt}'

        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ])

        try:
            response = await self.ollama_llm.ainvoke([message])
            raw_content = response.content

            # Robust JSON parsing
            start_index = raw_content.find('[')
            end_index = raw_content.rfind(']')
            if start_index != -1 and end_index != -1:
                json_string = raw_content[start_index : end_index + 1]
                return json.loads(json_string)
            else:
                logger.error("Could not find a valid JSON array in the LLM response.")
                return []
        except Exception as e:
            logger.error(f"LLM inference error during identification: {e}")
            raise HTTPException(status_code=503, detail="Error during food identification.")

    def _retrieve_and_calculate_nutrition(self, db_handler: DatabaseHandler, identified_items: list[dict]) -> dict:
        """Step 2: Look up facts in the DB, calculate totals, and format the final result."""
        total_calories = 0
        total_fat = 0.0
        total_saturated_fat = 0.0
        total_carbohydrates = 0.0
        total_sugar = 0.0
        total_protein = 0.0
        total_salt = 0.0

        # Determine the primary food item (the one with the largest quantity)
        primary_food_item_name = max(identified_items, key=lambda x: x.get('quantity_g', 0), default={}).get('item', 'N/A')

        for item in identified_items:
            food_name = item.get("item", "").lower()
            quantity_g = item.get("quantity_g", 0)

            if not food_name or not isinstance(quantity_g, int) or quantity_g <= 0:
                continue

            # Look up the food in our nutrition database
            fact = db_handler.find_nutrition_fact(food_name)
            if fact:
                # Calculate nutrition for the given quantity (facts are per 100g)
                multiplier = quantity_g / 100.0
                total_calories += fact.calories * multiplier
                total_fat += (fact.fat or 0) * multiplier
                total_saturated_fat += (fact.saturated_fat or 0) * multiplier
                total_carbohydrates += (fact.carbohydrates or 0) * multiplier
                total_sugar += (fact.sugar or 0) * multiplier
                total_protein += (fact.protein or 0) * multiplier
                total_salt += (fact.salt or 0) * multiplier
            else:
                logger.warning(f"No nutrition fact found in DB for item: '{food_name}'")

        # Format the final result in the structure the client expects
        return {
            "food_item": primary_food_item_name.title(),
            "calories": round(total_calories),
            "ingredients": {
                "fat": f"{total_fat:.1f}g",
                "saturated_fat": f"{total_saturated_fat:.1f}g",
                "carbohydrates": f"{total_carbohydrates:.1f}g",
                "sugar": f"{total_sugar:.1f}g",
                "protein": f"{total_protein:.1f}g",
                "salt": f"{total_salt:.0f}mg"
            }
        }

    async def _invoke_model(self, llm, image_bytes: bytes) -> str:
        """Helper function to invoke a LangChain model with multimodal input."""
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{encoded_image}"

        # Combine the user prompt with our structured analysis and security prompts
        final_prompt = f'{food_analysis_prompt} {security_prompt}'
        logger.debug("Using standardized prompt for analysis.")

        # LangChain's standard format for multimodal messages
        message = HumanMessage(
            content=[
                {"type": "text", "text": final_prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        )

        try:
            response = await llm.ainvoke([message])
            return response.content
        except Exception as e:
            logger.error("Inference service error: : %s", e)
            raise HTTPException(status_code=503, detail=f"Inference service error: {e}")

    async def analyze_image(self, image_bytes: bytes) -> dict:
        """Analyzes an image and robustly parses the JSON response."""
        logger.info("Using default Ollama inference engine for analysis.")
        raw_result = await self._invoke_model(self.ollama_llm, image_bytes)

        try:
            # --- ROBUST PARSING LOGIC ---
            # Find the first '{' and the last '}' in the raw response.
            start_index = raw_result.find('{')
            end_index = raw_result.rfind('}')

            if start_index != -1 and end_index != -1 and end_index > start_index:
                # Slice the string to get just the JSON part
                json_string = raw_result[start_index : end_index + 1]
                analysis_result = json.loads(json_string)
                return analysis_result
            else:
                # If we can't find a JSON object, raise an error.
                raise json.JSONDecodeError("Could not find a valid JSON object in the response.", raw_result, 0)

        except json.JSONDecodeError:
            logger.error("Failed to parse JSON from LLM response: %s", raw_result)
            return {
                "error": "Failed to parse analysis from the model.",
                "raw_response": raw_result
            }
        except Exception as e:
            logger.error("An unexpected error occurred during result parsing: %s", e)
            raise HTTPException(status_code=500, detail="Error processing the model's response.")

    async def analyze_image_with_ml_hybrid(self, image_bytes: bytes, user_query: str = None, mode: str = "hybrid") -> dict:
        """
        Hybrid analysis combining ML accuracy with LLaVA intelligence.

        Args:
            image_bytes: Image data
            user_query: Optional user question about the meal
            mode: Analysis mode - "ml_only", "hybrid", "llava_only"

        Returns:
            Combined analysis with accurate detection and intelligent insights
        """
        if mode == "llava_only":
            # Fallback to LLaVA only (original behavior)
            return await self.analyze_image(image_bytes)

        # Step 1: Get accurate ML analysis
        logger.info("Calling ML service for accurate food detection...")
        ml_results = await self._call_ml_service(image_bytes)

        if mode == "ml_only":
            # Return ML results only
            return {
                "mode": "ml_only",
                "detected_items": ml_results.get("food_items", []),
                "nutrition": ml_results.get("total_nutrition", {}),
                "source": "ml_service"
            }

        # Step 2: Enhance with LLaVA insights (hybrid mode)
        logger.info("Generating LLaVA insights based on ML results...")
        insights = await self._generate_llava_insights(image_bytes, ml_results, user_query)

        # Step 3: Combine results
        return {
            "mode": "hybrid",
            "detected_items": ml_results.get("food_items", []),
            "nutrition": ml_results.get("total_nutrition", {}),
            "insights": insights,
            "visualization_url": ml_results.get("visualization_url"),
            "sources": {
                "detection": "ml_service",
                "insights": "ollama_llava"
            }
        }

    async def analyze_image_with_db(self, db: Session, image_bytes: bytes, use_ml_service: bool = True) -> dict:
        """
        Orchestrates the full hybrid analysis process.

        Args:
            db: Database session
            image_bytes: Image data
            use_ml_service: If True, use ML service for accurate detection (recommended)
                           If False, use legacy LLaVA identification method

        Returns:
            Analysis result in the original format for backward compatibility
        """
        if use_ml_service:
            # NEW: Use ML service for accurate detection
            logger.info("Using ML service for accurate food detection...")
            try:
                ml_results = await self._call_ml_service(image_bytes)

                # Convert ML results to the format expected by the client
                # ML service already provides complete nutrition data
                food_items = ml_results.get("food_items", [])
                total_nutrition = ml_results.get("total_nutrition", {})

                if not food_items:
                    return {"error": "No food items detected in the image."}

                # Determine primary food item (largest by mass)
                primary_item = max(food_items, key=lambda x: x.get("estimated_mass_g", 0))

                # Format in the expected structure
                return {
                    "food_item": primary_item.get("item_name", "Unknown").title(),
                    "calories": int(total_nutrition.get("calories", 0)),
                    "ingredients": {
                        "fat": f"{total_nutrition.get('fat_g', 0):.1f}g",
                        "saturated_fat": f"{total_nutrition.get('saturated_fat_g', 0):.1f}g",
                        "carbohydrates": f"{total_nutrition.get('carb_g', 0):.1f}g",
                        "sugar": f"{total_nutrition.get('sugar_g', 0):.1f}g",
                        "protein": f"{total_nutrition.get('protein_g', 0):.1f}g",
                        "salt": f"{int(total_nutrition.get('sodium_mg', 0))}mg"
                    },
                    "source": "ml_service",
                    "all_items": food_items  # Additional data for client
                }
            except HTTPException as e:
                # If ML service fails, fall back to legacy method
                logger.warning(f"ML service failed ({e.detail}), falling back to LLaVA method")
                use_ml_service = False

        if not use_ml_service:
            # LEGACY: Use LLaVA to identify food items
            logger.info("Using legacy LLaVA identification method...")
            identified_items = await self._identify_food_items(image_bytes)

            if not identified_items:
                return {"error": "Could not identify any food items in the image."}

            # Use the database to get accurate nutrition data
            db_handler = DatabaseHandler(db)
            final_analysis = self._retrieve_and_calculate_nutrition(db_handler, identified_items)

            return final_analysis
# Create a single instance to be used across the application
inference_service = InferenceService()
