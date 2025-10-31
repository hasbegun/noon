from pydantic import BaseModel, Field
from typing import List

# Define the structure for a single ingredient/nutrient component
class FoodComponent(BaseModel):
    name: str = Field(description="Specific name of the ingredient or macronutrient, e.g., 'Chicken Breast', 'Olive Oil', 'Sugar'.")
    quantity_g: float = Field(description="Estimated mass of this component in grams.")
    calories: int = Field(description="Estimated calories (kcal) contributed by this component.")

# Define the main output structure for the entire meal
class MealAnalysis(BaseModel):
    dish_name: str = Field(description="The primary name of the dish, e.g., 'Chicken Caesar Salad' or 'Oatmeal with Walnuts'.")
    estimated_total_calories: int = Field(description="Total estimated calories (kcal) for the entire plate.")
    estimated_total_fat_g: float = Field(description="Total fat (grams) for the entire plate.")
    estimated_total_protein_g: float = Field(description="Total protein (grams) for the entire plate.")
    estimated_total_carb_g: float = Field(description="Total carbohydrates (grams) for the entire plate.")
    # Use FoodComponent for a list of detected ingredients/nutrients
    components: List[FoodComponent] = Field(description="A list of 3-5 major ingredients or calorie contributors.")
    notes: str = Field(description="Brief comments on portion size or missing visible ingredients (e.g., 'dressing not accounted for').")