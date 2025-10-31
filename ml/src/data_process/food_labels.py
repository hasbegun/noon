"""
Food category label mappings for different datasets

This module manages food category labels across different datasets
and provides unified label mapping functionality.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger


class FoodLabelManager:
    """Manages food category labels across datasets"""

    # Food-101 categories (101 classes)
    FOOD101_CATEGORIES = [
        'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
        'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
        'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
        'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla',
        'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
        'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
        'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
        'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras',
        'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
        'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
        'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
        'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
        'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup',
        'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters',
        'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
        'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
        'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto',
        'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits',
        'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
        'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare',
        'waffles'
    ]

    # Nutrition5k food categories (these are dish types from the dataset)
    NUTRITION5K_CATEGORIES = [
        'rice', 'chicken', 'beef', 'pork', 'fish', 'vegetables', 'salad',
        'pasta', 'bread', 'soup', 'sandwich', 'burger', 'pizza', 'fries',
        'dessert', 'fruit', 'mixed_dish', 'other'
    ]

    def __init__(self, dataset_name: str = "food-101"):
        """
        Initialize label manager

        Args:
            dataset_name: Name of dataset ('food-101', 'nutrition5k', or 'combined')
        """
        self.dataset_name = dataset_name

        if dataset_name == "food-101":
            self.categories = self.FOOD101_CATEGORIES
        elif dataset_name == "nutrition5k":
            self.categories = self.NUTRITION5K_CATEGORIES
        elif dataset_name == "combined":
            # Combine all categories and remove duplicates
            all_categories = set(self.FOOD101_CATEGORIES + self.NUTRITION5K_CATEGORIES)
            self.categories = sorted(list(all_categories))
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Create mappings
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.categories)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        logger.info(f"Initialized FoodLabelManager for {dataset_name}: {len(self.categories)} categories")

    @property
    def num_classes(self) -> int:
        """Get number of classes"""
        return len(self.categories)

    def get_class_name(self, idx: int) -> str:
        """Get class name from index"""
        return self.idx_to_class.get(idx, "unknown")

    def get_class_idx(self, name: str) -> int:
        """Get class index from name"""
        return self.class_to_idx.get(name, -1)

    def save_mapping(self, path: Path):
        """Save label mapping to JSON file"""
        path.parent.mkdir(parents=True, exist_ok=True)

        mapping = {
            'dataset': self.dataset_name,
            'num_classes': self.num_classes,
            'categories': self.categories,
            'class_to_idx': self.class_to_idx,
            'idx_to_class': {str(k): v for k, v in self.idx_to_class.items()},
        }

        with open(path, 'w') as f:
            json.dump(mapping, f, indent=2)

        logger.info(f"Label mapping saved to {path}")

    @classmethod
    def load_mapping(cls, path: Path) -> "FoodLabelManager":
        """Load label mapping from JSON file"""
        with open(path, 'r') as f:
            mapping = json.load(f)

        manager = cls(dataset_name=mapping['dataset'])
        return manager

    def get_readable_name(self, class_name: str) -> str:
        """
        Convert class name to human-readable format

        Examples:
            'apple_pie' -> 'Apple Pie'
            'chicken_quesadilla' -> 'Chicken Quesadilla'
        """
        return class_name.replace('_', ' ').title()

    def normalize_name(self, name: str) -> str:
        """
        Normalize food name to match class names

        Examples:
            'Apple Pie' -> 'apple_pie'
            'chicken-quesadilla' -> 'chicken_quesadilla'
        """
        return name.lower().replace(' ', '_').replace('-', '_')

    def find_similar_classes(self, query: str, top_k: int = 5) -> List[str]:
        """
        Find similar class names using simple string matching

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of similar class names
        """
        query_normalized = self.normalize_name(query)

        # Calculate similarity scores
        scores = []
        for cls in self.categories:
            # Simple substring matching
            if query_normalized in cls:
                scores.append((cls, 1.0))
            elif cls in query_normalized:
                scores.append((cls, 0.8))
            else:
                # Count common words
                query_words = set(query_normalized.split('_'))
                cls_words = set(cls.split('_'))
                common = len(query_words & cls_words)
                if common > 0:
                    scores.append((cls, common / max(len(query_words), len(cls_words))))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        return [cls for cls, _ in scores[:top_k]]


# Create singleton instances for convenience
food101_labels = FoodLabelManager("food-101")
nutrition5k_labels = FoodLabelManager("nutrition5k")


def get_label_manager(dataset_name: str) -> FoodLabelManager:
    """
    Get label manager for a specific dataset

    Args:
        dataset_name: Dataset name

    Returns:
        FoodLabelManager instance
    """
    if dataset_name == "food-101":
        return food101_labels
    elif dataset_name == "nutrition5k":
        return nutrition5k_labels
    else:
        return FoodLabelManager(dataset_name)


if __name__ == "__main__":
    # Test label manager
    manager = FoodLabelManager("food-101")
    print(f"Number of classes: {manager.num_classes}")
    print(f"First 10 categories: {manager.categories[:10]}")
    print(f"Class 'pizza' -> idx {manager.get_class_idx('pizza')}")
    print(f"Idx 0 -> class '{manager.get_class_name(0)}'")
    print(f"Readable name: {manager.get_readable_name('apple_pie')}")
    print(f"Similar to 'pizza': {manager.find_similar_classes('pizza')}")
