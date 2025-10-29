"""
USDA FoodData Central lookup service
Provides nutrition information for detected food items
"""
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger
from sqlalchemy import Column, Float, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from ..config import config

Base = declarative_base()


class FoodItem(Base):
    """SQLAlchemy model for food items"""

    __tablename__ = "foods"

    id = Column(Integer, primary_key=True)
    fdc_id = Column(Integer, unique=True, index=True)
    description = Column(String, index=True)
    data_type = Column(String)
    food_category = Column(String, index=True)

    # Macronutrients (per 100g)
    energy_kcal = Column(Float)
    protein_g = Column(Float)
    carbohydrate_g = Column(Float)
    fat_g = Column(Float)
    fiber_g = Column(Float)
    sugar_g = Column(Float)

    # Minerals
    sodium_mg = Column(Float)
    calcium_mg = Column(Float)
    iron_mg = Column(Float)
    potassium_mg = Column(Float)
    magnesium_mg = Column(Float)
    phosphorus_mg = Column(Float)

    # Vitamins
    vitamin_a_ug = Column(Float)
    vitamin_c_mg = Column(Float)
    vitamin_d_ug = Column(Float)
    vitamin_e_mg = Column(Float)

    # Additional
    cholesterol_mg = Column(Float)
    saturated_fat_g = Column(Float)

    # Raw JSON for full details
    nutrients_json = Column(Text)


class USDALookupService:
    """Service for looking up nutrition information from USDA data"""

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize USDA lookup service

        Args:
            db_path: Path to SQLite database (will be created if not exists)
        """
        self.db_path = db_path or config.usda_db_path
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Create tables if they don't exist
        if not self.db_path.exists():
            logger.info("Initializing USDA database")
            self._initialize_database()

    def _initialize_database(self):
        """Initialize database from JSON files"""
        Base.metadata.create_all(self.engine)

        # Load USDA JSON files
        usda_files = [
            config.usda_data_path / config.usda_foundation_json,
            config.usda_data_path / config.usda_branded_json,
        ]

        with self.SessionLocal() as session:
            for usda_file in usda_files:
                if usda_file.exists():
                    logger.info(f"Loading {usda_file.name}")
                    self._load_usda_json(usda_file, session)
                else:
                    logger.warning(f"File not found: {usda_file}")

            session.commit()

        logger.info("USDA database initialized")

    def _load_usda_json(self, json_file: Path, session: Session):
        """Load USDA JSON data into database"""
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Determine data type
            if "FoundationFoods" in data:
                foods = data["FoundationFoods"]
                data_type = "Foundation"
            elif "BrandedFoods" in data:
                foods = data["BrandedFoods"]
                data_type = "Branded"
            elif "SRLegacyFoods" in data:
                foods = data["SRLegacyFoods"]
                data_type = "SRLegacy"
            else:
                logger.warning(f"Unknown data format in {json_file}")
                return

            # Process foods
            for food in foods:
                self._add_food_item(food, data_type, session)

        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")

    def _add_food_item(self, food: Dict, data_type: str, session: Session):
        """Add a food item to the database"""
        try:
            # Extract basic info
            fdc_id = food.get("fdcId")
            description = food.get("description", "")

            # Check if already exists
            existing = session.query(FoodItem).filter_by(fdc_id=fdc_id).first()
            if existing:
                return

            # Extract nutrients
            nutrients = self._extract_nutrients(food.get("foodNutrients", []))

            # Create food item
            food_item = FoodItem(
                fdc_id=fdc_id,
                description=description,
                data_type=data_type,
                food_category=food.get("foodCategory", {}).get("description", ""),
                **nutrients,
                nutrients_json=json.dumps(food.get("foodNutrients", []))
            )

            session.add(food_item)

        except Exception as e:
            logger.error(f"Error adding food item: {e}")

    def _extract_nutrients(self, nutrients_list: List[Dict]) -> Dict[str, Optional[float]]:
        """Extract key nutrients from USDA nutrient list"""
        nutrient_map = {
            "Energy": "energy_kcal",
            "Protein": "protein_g",
            "Carbohydrate, by difference": "carbohydrate_g",
            "Total lipid (fat)": "fat_g",
            "Fiber, total dietary": "fiber_g",
            "Sugars, total including NLEA": "sugar_g",
            "Sodium, Na": "sodium_mg",
            "Calcium, Ca": "calcium_mg",
            "Iron, Fe": "iron_mg",
            "Potassium, K": "potassium_mg",
            "Magnesium, Mg": "magnesium_mg",
            "Phosphorus, P": "phosphorus_mg",
            "Vitamin A, RAE": "vitamin_a_ug",
            "Vitamin C, total ascorbic acid": "vitamin_c_mg",
            "Vitamin D (D2 + D3)": "vitamin_d_ug",
            "Vitamin E (alpha-tocopherol)": "vitamin_e_mg",
            "Cholesterol": "cholesterol_mg",
            "Fatty acids, total saturated": "saturated_fat_g",
        }

        result = {v: None for v in nutrient_map.values()}

        for nutrient in nutrients_list:
            nutrient_name = nutrient.get("nutrient", {}).get("name", "")
            amount = nutrient.get("amount")

            if nutrient_name in nutrient_map and amount is not None:
                key = nutrient_map[nutrient_name]

                # Convert energy from kJ to kcal if needed
                if key == "energy_kcal":
                    unit = nutrient.get("nutrient", {}).get("unitName", "kcal")
                    if unit.lower() == "kj":
                        amount = amount / 4.184  # kJ to kcal

                result[key] = float(amount)

        return result

    def search(
        self,
        query: str,
        limit: int = 10,
        category: Optional[str] = None,
    ) -> List[Dict]:
        """
        Search for food items

        Args:
            query: Search query
            limit: Maximum number of results
            category: Optional food category filter

        Returns:
            List of matching food items
        """
        with self.SessionLocal() as session:
            q = session.query(FoodItem)

            # Text search
            if query:
                search_term = f"%{query}%"
                q = q.filter(FoodItem.description.ilike(search_term))

            # Category filter
            if category:
                q = q.filter(FoodItem.food_category.ilike(f"%{category}%"))

            # Get results
            results = q.limit(limit).all()

            return [self._food_item_to_dict(item) for item in results]

    def get_by_id(self, fdc_id: int) -> Optional[Dict]:
        """Get food item by FDC ID"""
        with self.SessionLocal() as session:
            item = session.query(FoodItem).filter_by(fdc_id=fdc_id).first()
            if item:
                return self._food_item_to_dict(item)
            return None

    def get_nutrition_for_portion(
        self,
        fdc_id: int,
        portion_g: float,
    ) -> Optional[Dict]:
        """
        Get nutrition information scaled to portion size

        Args:
            fdc_id: Food item ID
            portion_g: Portion size in grams

        Returns:
            Nutrition information for the portion
        """
        food_item = self.get_by_id(fdc_id)
        if not food_item:
            return None

        # USDA data is per 100g, scale to portion
        scale_factor = portion_g / 100.0

        nutrition = {}
        for key, value in food_item.items():
            if key.startswith(("energy_", "protein_", "carbohydrate_", "fat_",
                             "fiber_", "sugar_", "sodium_", "calcium_",
                             "iron_", "potassium_", "vitamin_", "cholesterol_",
                             "saturated_", "magnesium_", "phosphorus_")):
                if value is not None:
                    nutrition[key] = value * scale_factor
                else:
                    nutrition[key] = None

        nutrition["description"] = food_item["description"]
        nutrition["portion_g"] = portion_g
        nutrition["fdc_id"] = fdc_id

        return nutrition

    def find_best_match(
        self,
        food_name: str,
        category_hint: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Find best matching food item

        Args:
            food_name: Name/description of food
            category_hint: Optional category hint

        Returns:
            Best matching food item
        """
        results = self.search(food_name, limit=5, category=category_hint)

        if not results:
            return None

        # Return first result (can be improved with better ranking)
        return results[0]

    def _food_item_to_dict(self, item: FoodItem) -> Dict:
        """Convert FoodItem to dictionary"""
        return {
            "fdc_id": item.fdc_id,
            "description": item.description,
            "data_type": item.data_type,
            "food_category": item.food_category,
            "energy_kcal": item.energy_kcal,
            "protein_g": item.protein_g,
            "carbohydrate_g": item.carbohydrate_g,
            "fat_g": item.fat_g,
            "fiber_g": item.fiber_g,
            "sugar_g": item.sugar_g,
            "sodium_mg": item.sodium_mg,
            "calcium_mg": item.calcium_mg,
            "iron_mg": item.iron_mg,
            "potassium_mg": item.potassium_mg,
            "magnesium_mg": item.magnesium_mg,
            "phosphorus_mg": item.phosphorus_mg,
            "vitamin_a_ug": item.vitamin_a_ug,
            "vitamin_c_mg": item.vitamin_c_mg,
            "vitamin_d_ug": item.vitamin_d_ug,
            "vitamin_e_mg": item.vitamin_e_mg,
            "cholesterol_mg": item.cholesterol_mg,
            "saturated_fat_g": item.saturated_fat_g,
        }
