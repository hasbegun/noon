#!/usr/bin/env python3
"""
Quick test of placeholder SAM2 model
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from src.models.sam2_segmentation import SAM2Segmentor

print("Testing placeholder SAM2 model...")

try:
    # Create segmentor (will use placeholder since SAM2 config not found)
    segmentor = SAM2Segmentor(device="cpu")
    print("✓ Segmentor created successfully")

    # Test with dummy image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print("✓ Created test image")

    # Test automatic segmentation
    masks = segmentor.segment_automatic(image)
    print(f"✓ Automatic segmentation returned {len(masks)} masks")

    # Test point-based segmentation
    point_coords = np.array([[320, 240]])
    point_labels = np.array([1])
    masks, scores, logits = segmentor.segment_with_points(image, point_coords, point_labels)
    print(f"✓ Point-based segmentation returned {len(masks)} masks")

    print("\n✅ All tests passed! Placeholder model is working correctly.")

except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
