#!/usr/bin/env python3
"""
Check SAM2 installation status
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.models.sam2_segmentation import SAM2Segmentor

    seg = SAM2Segmentor(device='cpu')

    if hasattr(seg, 'use_placeholder') and seg.use_placeholder:
        print('\033[0;33m⚠ Using placeholder model\033[0m')
        print('\033[0;33m  Install SAM2: make install-sam2\033[0m')
        print('\033[0;33m  Download checkpoints: make download-sam2-checkpoints\033[0m')
        sys.exit(1)  # Exit with error code to indicate placeholder
    else:
        print('\033[0;32m✓ Real SAM2 model is working!\033[0m')
        sys.exit(0)

except Exception as e:
    print(f'\033[0;31m✗ Error: {e}\033[0m')
    sys.exit(2)
