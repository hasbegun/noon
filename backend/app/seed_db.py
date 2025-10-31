import logging
from storage_service import SessionLocal
from db_classes import NutritionFact

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample nutritional data (per 100g)
# In a real application, this would come from a CSV, an API, or a larger dataset.
NUTRITION_DATA = [
    {"name": "chicken breast", "calories": 165, "fat": 3.6, "saturated_fat": 1, "carbohydrates": 0, "sugar": 0, "protein": 31, "salt": 74},
    {"name": "romaine lettuce", "calories": 17, "fat": 0.3, "saturated_fat": 0, "carbohydrates": 3.3, "sugar": 1.2, "protein": 1.2, "salt": 8},
    {"name": "cherry tomato", "calories": 18, "fat": 0.2, "saturated_fat": 0, "carbohydrates": 3.9, "sugar": 2.6, "protein": 0.9, "salt": 5},
    {"name": "croutons", "calories": 407, "fat": 9, "saturated_fat": 1.8, "carbohydrates": 72, "sugar": 7, "protein": 11, "salt": 766},
    {"name": "cheeseburger", "calories": 303, "fat": 14, "saturated_fat": 6, "carbohydrates": 28, "sugar": 5, "protein": 17, "salt": 629},
    {"name": "french fries", "calories": 312, "fat": 15, "saturated_fat": 2.3, "carbohydrates": 41, "sugar": 0.3, "protein": 3.4, "salt": 210},
    {"name": "broccoli", "calories": 55, "fat": 0.6, "saturated_fat": 0.1, "carbohydrates": 11.2, "sugar": 2.7, "protein": 3.7, "salt": 33},
    {"name": "salmon fillet", "calories": 208, "fat": 13, "saturated_fat": 3, "carbohydrates": 0, "sugar": 0, "protein": 20, "salt": 59},
]

def seed_nutrition_data():
    """Populates the nutrition_facts table with sample data."""
    db = SessionLocal()
    try:
        existing_items = {item.name for item in db.query(NutritionFact).all()}

        new_items_to_add = []
        for item_data in NUTRITION_DATA:
            if item_data["name"] not in existing_items:
                new_items_to_add.append(NutritionFact(**item_data))

        if not new_items_to_add:
            logger.info("Nutrition data already seeded. No new items to add.")
            return

        db.add_all(new_items_to_add)
        db.commit()
        logger.info(f"Successfully seeded {len(new_items_to_add)} new nutrition facts into the database.")

    except Exception as e:
        logger.error(f"Failed to seed database: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    logger.info("Starting database seeding process for nutrition facts...")
    seed_nutrition_data()