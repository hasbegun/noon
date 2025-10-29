#!/usr/bin/env python3
"""
Inference script for food detection and nutrition analysis
"""
import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from loguru import logger

from src.config import config
from src.models import FoodDetector
from src.services import NutritionAnalyzer, USDALookupService


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run inference on food images")

    # Input/Output
    parser.add_argument("--image", type=Path, required=True, help="Path to input image")
    parser.add_argument("--output", type=Path, default=None, help="Path to save results")
    parser.add_argument("--save-viz", action="store_true", help="Save visualization images")

    # Food labels
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Comma-separated food labels (e.g., 'rice,chicken,salad')",
    )

    # Model
    parser.add_argument(
        "--model-type",
        type=str,
        default=config.sam2_model_type,
        choices=["vit_b", "vit_l", "vit_h"],
        help="SAM2 model type",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Model checkpoint")

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=config.device,
        choices=["cuda", "mps", "cpu"],
        help="Device to use",
    )

    # Analysis options
    parser.add_argument("--detect-only", action="store_true", help="Only run detection, skip nutrition")

    return parser.parse_args()


def main():
    """Main inference function"""
    args = parse_args()

    # Validate input
    if not args.image.exists():
        logger.error(f"Image not found: {args.image}")
        return

    logger.info("Food Detection & Nutrition Analysis")
    logger.info(f"Input image: {args.image}")

    # Initialize detector
    logger.info(f"Loading model: {args.model_type}")
    detector = FoodDetector(
        sam2_model_type=args.model_type,
        sam2_checkpoint=args.checkpoint,
        device=args.device,
    )

    # Parse food labels
    food_labels = None
    if args.labels:
        food_labels = [label.strip() for label in args.labels.split(",")]
        logger.info(f"Food labels: {food_labels}")

    try:
        if args.detect_only:
            # Detection only
            logger.info("Running food detection...")
            results = detector.detect_and_analyze(
                str(args.image),
                return_visualizations=args.save_viz,
            )

            logger.info(f"Detected {results['num_items']} food items")

            # Print results
            for i, item in enumerate(results["food_items"]):
                logger.info(f"Item {i + 1}:")
                logger.info(f"  Volume: {item['volume_ml']:.1f} ml")
                logger.info(f"  Area: {item['area_cm2']:.1f} cm²")
                logger.info(f"  Confidence: {item['confidence']:.2f}")

        else:
            # Full nutrition analysis
            logger.info("Running nutrition analysis...")

            # Initialize services
            usda_service = USDALookupService()
            analyzer = NutritionAnalyzer(detector=detector, usda_service=usda_service)

            # Run analysis
            results = analyzer.analyze_image(
                str(args.image),
                food_labels=food_labels,
                return_visualization=args.save_viz,
            )

            logger.info(f"Analyzed {results['total_items']} food items")

            # Print results
            print("\n" + "=" * 60)
            print("NUTRITION ANALYSIS RESULTS")
            print("=" * 60)

            for i, item in enumerate(results["items"]):
                print(f"\nItem {i + 1}: {item['food_label']}")
                print(f"  Volume: {item['volume_ml']:.1f} ml")
                print(f"  Mass: {item['estimated_mass_g']:.1f} g")
                print(f"  Confidence: {item['confidence']:.2f}")

                if "nutrition" in item and "energy_kcal" in item["nutrition"]:
                    nutrition = item["nutrition"]
                    print("  Nutrition:")
                    print(f"    Calories: {nutrition.get('energy_kcal', 0):.0f} kcal")
                    print(f"    Protein: {nutrition.get('protein_g', 0):.1f} g")
                    print(f"    Carbs: {nutrition.get('carbohydrate_g', 0):.1f} g")
                    print(f"    Fat: {nutrition.get('fat_g', 0):.1f} g")
                    print(f"    Fiber: {nutrition.get('fiber_g', 0):.1f} g")
                    print(f"    Sodium: {nutrition.get('sodium_mg', 0):.0f} mg")

            # Print totals
            totals = results["totals"]
            print("\n" + "=" * 60)
            print("TOTALS")
            print("=" * 60)
            print(f"Total Volume: {totals['total_volume_ml']:.1f} ml")
            print(f"Total Mass: {totals['total_mass_g']:.1f} g")
            print(f"Total Calories: {totals['total_energy_kcal']:.0f} kcal")
            print(f"Total Protein: {totals['total_protein_g']:.1f} g")
            print(f"Total Carbs: {totals['total_carbohydrate_g']:.1f} g")
            print(f"Total Fat: {totals['total_fat_g']:.1f} g")
            print(f"Total Sodium: {totals['total_sodium_mg']:.0f} mg")
            print("=" * 60 + "\n")

        # Save results
        if args.output:
            output_dir = args.output.parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save JSON results with input filename
            json_filename = f"{args.image.stem}_result.json"
            json_file = output_dir / json_filename
            with open(json_file, "w") as f:
                # Remove non-serializable items
                results_copy = {
                    "input_image": str(args.image.name),
                    "num_items": results["num_items"],
                    "image_shape": list(results["image_shape"]),
                    "food_items": []
                }

                # Clean food items for JSON serialization
                for item in results.get("food_items", []):
                    item_copy = {
                        "item_id": item["item_id"],
                        "item_name": item.get("item_name", f"Food Item {item['item_id'] + 1}"),
                        "area_pixels": int(item["area_pixels"]),
                        "bbox": list(item["bbox"]) if not isinstance(item["bbox"], list) else item["bbox"],
                        "predicted_iou": float(item.get("predicted_iou", 0.0)),
                        "volume_ml": float(item.get("volume_ml", 0.0)),
                        "volume_cm3": float(item.get("volume_cm3", 0.0)),
                        "area_cm2": float(item.get("area_cm2", 0.0)),
                        "estimated_height_cm": float(item.get("estimated_height_cm", 0.0)),
                        "circularity": float(item.get("circularity", 0.0)),
                        "compactness": float(item.get("compactness", 0.0)),
                        "confidence": float(item.get("confidence", 0.0)),
                    }
                    results_copy["food_items"].append(item_copy)

                json.dump(results_copy, f, indent=2)

            logger.info(f"Results saved to: {json_file}")

            # Save visualizations
            if args.save_viz and "visualizations" in results:
                viz_dir = output_dir / "visualizations"
                viz_dir.mkdir(exist_ok=True)

                for viz_name, viz_image in results["visualizations"].items():
                    viz_file = viz_dir / f"{args.image.stem}_{viz_name}.jpg"
                    cv2.imwrite(str(viz_file), cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR))
                    logger.info(f"Visualization saved: {viz_file}")

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


if __name__ == "__main__":
    main()
