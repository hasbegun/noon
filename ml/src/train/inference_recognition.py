#!/usr/bin/env python3
"""
Inference script for SAM2 + Food Recognition pipeline

This script demonstrates the complete pipeline:
1. SAM2 segments food items in an image
2. Food Recognition model classifies each detected region
3. USDA lookup provides nutrition information

Usage:
    python src/train/inference_recognition.py \
        --image path/to/food_image.jpg \
        --model-path models/recognition/food-101_efficientnet_b0/best_f1.pt \
        --output results/

    # With visualization
    python src/train/inference_recognition.py \
        --image tests/test_food1.jpg \
        --model-path models/recognition/food-101_efficientnet_b0/best_f1.pt \
        --visualize \
        --output results/
"""
import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image

from data_process.food_labels import FoodLabelManager
from models import FoodRecognitionModel, SAM2Segmentor
from services import USDALookupService


class FoodRecognitionPipeline:
    """Complete pipeline: SAM2 segmentation + food recognition + nutrition lookup"""

    def __init__(
        self,
        recognition_model_path: Path,
        label_mapping_path: Path,
        sam2_model_type: str = "hiera_b+",
        device: str = "mps",
    ):
        """
        Initialize the pipeline

        Args:
            recognition_model_path: Path to trained recognition model
            label_mapping_path: Path to label mapping JSON
            sam2_model_type: SAM2 model variant
            device: Device to use
        """
        self.device = device

        # Load label manager
        logger.info(f"Loading label mapping from {label_mapping_path}")
        self.label_manager = FoodLabelManager.load_mapping(label_mapping_path)

        # Load recognition model
        logger.info(f"Loading recognition model from {recognition_model_path}")
        self.recognition_model = FoodRecognitionModel.load(
            recognition_model_path,
            device=device
        )
        self.recognition_model.eval()

        # Initialize SAM2 segmentor
        logger.info(f"Initializing SAM2 segmentor: {sam2_model_type}")
        self.sam2 = SAM2Segmentor(model_type=sam2_model_type)

        # Initialize USDA lookup service
        logger.info("Initializing USDA nutrition lookup")
        self.usda_service = USDALookupService()

        logger.info("Pipeline initialized successfully!")

    def process_image(
        self,
        image_path: Path,
        confidence_threshold: float = 0.5,
        min_region_size: int = 1000,
    ) -> dict:
        """
        Process an image through the complete pipeline

        Args:
            image_path: Path to input image
            confidence_threshold: Minimum confidence for predictions
            min_region_size: Minimum region size in pixels

        Returns:
            Dictionary with detection results
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Step 1: Segment with SAM2
        logger.info("Step 1: Segmenting image with SAM2...")
        masks = self.sam2.segment_automatic(image_rgb)

        if not masks or len(masks) == 0:
            logger.warning("No regions detected by SAM2")
            return {
                "image_path": str(image_path),
                "detections": [],
                "num_detections": 0,
            }

        logger.info(f"SAM2 detected {len(masks)} regions")

        # Step 2: Filter masks by size
        filtered_masks = []
        for mask in masks:
            mask_area = mask['segmentation'].sum()
            if mask_area >= min_region_size:
                filtered_masks.append(mask)

        logger.info(f"After filtering: {len(filtered_masks)} regions")

        if len(filtered_masks) == 0:
            return {
                "image_path": str(image_path),
                "detections": [],
                "num_detections": 0,
            }

        # Step 3: Extract crops and classify
        logger.info("Step 2: Classifying detected regions...")
        detections = []

        for i, mask_data in enumerate(filtered_masks):
            mask = mask_data['segmentation']

            # Get bounding box
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0:
                continue

            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()

            # Extract crop
            crop = image_rgb[y_min:y_max, x_min:x_max]

            if crop.size == 0:
                continue

            # Preprocess crop for recognition model
            crop_tensor = self._preprocess_crop(crop)

            # Classify
            with torch.no_grad():
                logits = self.recognition_model(crop_tensor)
                probs = F.softmax(logits, dim=1)
                confidence, pred_idx = torch.max(probs, dim=1)

            pred_idx = pred_idx.item()
            confidence = confidence.item()

            # Filter by confidence
            if confidence < confidence_threshold:
                continue

            # Get class name
            class_name = self.label_manager.get_class_name(pred_idx)
            readable_name = self.label_manager.get_readable_name(class_name)

            # Get bounding box coordinates
            bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]

            # Estimate portion size (very rough)
            mask_area_pixels = mask.sum()
            estimated_mass_g = self._estimate_mass(mask_area_pixels, class_name)

            detection = {
                "region_id": i,
                "food_class": class_name,
                "food_name": readable_name,
                "confidence": float(confidence),
                "bbox": bbox,
                "mask_area_pixels": int(mask_area_pixels),
                "estimated_mass_g": estimated_mass_g,
            }

            detections.append(detection)

        logger.info(f"Classified {len(detections)} food items (confidence >= {confidence_threshold})")

        # Step 4: Lookup nutrition information
        logger.info("Step 3: Looking up nutrition information...")
        for detection in detections:
            nutrition = self._lookup_nutrition(
                detection["food_name"],
                detection["estimated_mass_g"]
            )
            detection["nutrition"] = nutrition

        # Prepare result
        result = {
            "image_path": str(image_path),
            "detections": detections,
            "num_detections": len(detections),
        }

        return result

    def _preprocess_crop(self, crop: np.ndarray) -> torch.Tensor:
        """Preprocess image crop for recognition model"""
        from data_process.classification_dataset import FoodClassificationDataset
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        # Use same preprocessing as training
        transform = A.Compose([
            A.Resize(224, 224),  # Standard ImageNet size
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])

        transformed = transform(image=crop)
        image_tensor = transformed["image"].unsqueeze(0).to(self.device)

        return image_tensor

    def _estimate_mass(self, mask_area_pixels: int, food_class: str) -> float:
        """
        Estimate mass in grams from mask area

        This is a very rough estimation. In a real system, you would use:
        - Depth information
        - Reference object size
        - Food density database
        """
        # Very rough heuristic: assume 1000 pixels ≈ 50g
        # This will be wildly inaccurate but gives a starting point
        estimated_mass = (mask_area_pixels / 1000.0) * 50.0

        # Clamp to reasonable range
        estimated_mass = max(10.0, min(estimated_mass, 500.0))

        return estimated_mass

    def _lookup_nutrition(self, food_name: str, portion_g: float) -> dict:
        """Lookup nutrition information from USDA database"""
        try:
            # Search for food
            search_results = self.usda_service.search_food(food_name, limit=1)

            if not search_results:
                return None

            # Get nutrition for portion
            food_id = search_results[0]["fdc_id"]
            nutrition = self.usda_service.get_nutrition(food_id, portion_g)

            return nutrition

        except Exception as e:
            logger.warning(f"Failed to lookup nutrition for '{food_name}': {e}")
            return None

    def visualize_results(
        self,
        image_path: Path,
        detections: list,
        output_path: Path
    ):
        """Create visualization with bounding boxes and labels"""
        image = cv2.imread(str(image_path))

        for detection in detections:
            bbox = detection["bbox"]
            food_name = detection["food_name"]
            confidence = detection["confidence"]

            # Draw bounding box
            cv2.rectangle(
                image,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                (0, 255, 0),
                2
            )

            # Add label
            label = f"{food_name}: {confidence:.2f}"
            cv2.putText(
                image,
                label,
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        # Save visualization
        cv2.imwrite(str(output_path), image)
        logger.info(f"Visualization saved to {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Food recognition inference")

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained recognition model (.pt file)",
    )
    parser.add_argument(
        "--label-mapping",
        type=Path,
        default=None,
        help="Path to label mapping JSON (default: same dir as model)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Output directory",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualization",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["cuda", "mps", "cpu"],
        help="Device to use",
    )

    args = parser.parse_args()

    # Setup output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Auto-detect label mapping if not provided
    if args.label_mapping is None:
        args.label_mapping = args.model_path.parent / "label_mapping.json"

    if not args.label_mapping.exists():
        logger.error(f"Label mapping not found: {args.label_mapping}")
        logger.error("Please provide --label-mapping path")
        return

    # Initialize pipeline
    logger.info("Initializing Food Recognition Pipeline...")
    pipeline = FoodRecognitionPipeline(
        recognition_model_path=args.model_path,
        label_mapping_path=args.label_mapping,
        device=args.device,
    )

    # Process image
    logger.info(f"\nProcessing image: {args.image}")
    results = pipeline.process_image(
        args.image,
        confidence_threshold=args.confidence,
    )

    # Print results
    logger.info(f"\n{'='*60}")
    logger.info(f"Results: {results['num_detections']} food items detected")
    logger.info(f"{'='*60}")

    for detection in results['detections']:
        logger.info(f"\n{detection['food_name']}:")
        logger.info(f"  Confidence: {detection['confidence']:.2%}")
        logger.info(f"  Estimated mass: {detection['estimated_mass_g']:.1f}g")

        if detection.get('nutrition'):
            nutrition = detection['nutrition']
            logger.info(f"  Nutrition (per {nutrition['portion_g']:.0f}g):")
            logger.info(f"    Calories: {nutrition['calories']:.1f} kcal")
            logger.info(f"    Protein: {nutrition['protein_g']:.1f}g")
            logger.info(f"    Carbs: {nutrition['carbohydrates_g']:.1f}g")
            logger.info(f"    Fat: {nutrition['fat_g']:.1f}g")

    # Save results
    results_file = args.output / f"{args.image.stem}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {results_file}")

    # Create visualization
    if args.visualize:
        viz_file = args.output / f"{args.image.stem}_visualization.jpg"
        pipeline.visualize_results(args.image, results['detections'], viz_file)

    logger.info(f"\n{'='*60}")
    logger.info("✓ Inference complete!")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
