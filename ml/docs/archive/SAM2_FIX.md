# SAM2 Placeholder Model Fix

## Problem

When SAM2 fails to load (config not found or checkpoint missing), the system tried to use a placeholder model but crashed with:
```
AttributeError: 'PlaceholderSAM2' object has no attribute 'image_size'
```

The issue was that `SAM2ImagePredictor` expected certain attributes on the model that the placeholder didn't have.

## Solution

Created a complete placeholder implementation that mimics SAM2's interface:

### 1. Enhanced PlaceholderSAM2 Model

Added required attributes:
```python
class PlaceholderSAM2(nn.Module):
    def __init__(self, image_size=1024):
        super().__init__()
        self.image_size = image_size  # ✓ Required by SAM2ImagePredictor
        self.image_encoder = None      # ✓ Placeholder
        self.mask_decoder = None       # ✓ Placeholder
        # ... actual model code ...
```

### 2. Created PlaceholderPredictor

Instead of using `SAM2ImagePredictor` with placeholder model, created a compatible predictor:

```python
class PlaceholderPredictor:
    def set_image(self, image): ...
    def predict(self, point_coords, point_labels, box, multimask_output): ...
    def generate(self, points_per_side, pred_iou_thresh, stability_score_thresh): ...
    def _simple_segmentation(self): ...
```

### 3. Conditional Predictor Selection

```python
if hasattr(self, 'use_placeholder') and self.use_placeholder:
    self.predictor = self._create_placeholder_predictor()
else:
    self.predictor = SAM2ImagePredictor(self.sam2_model)
```

## How It Works

### Placeholder Segmentation

Uses traditional computer vision for basic segmentation:

1. **Automatic Segmentation**:
   - Converts to grayscale
   - Applies binary threshold
   - Finds contours
   - Filters by area
   - Returns masks with metadata

2. **Point-based Segmentation**:
   - Creates circular regions around points
   - Marks foreground/background
   - Returns multiple masks with scores

3. **Box-based Segmentation**:
   - Creates masks within bounding boxes
   - Returns mask and confidence score

## When Does It Activate?

Placeholder activates when:
- ❌ SAM2 config file not found
- ❌ SAM2 checkpoint not found
- ❌ SAM2 installation failed
- ❌ Any error during SAM2 model building

## Testing

Test the placeholder model:

```bash
# Test with Python
python test_placeholder.py

# Test with inference script (will use placeholder if SAM2 not available)
python scripts/inference.py \
    --image test-food1.jpg \
    --detect-only \
    --save-viz \
    --output results/
```

## Getting Real SAM2

To use the actual SAM2 model:

### Option 1: Install SAM2 Properly

```bash
# Make sure SAM2 is installed
make install-sam2

# Or manual
mkdir -p .tmp
cd .tmp
git clone https://github.com/facebookresearch/sam2.git
cd sam2
conda run -n noon2 pip install -e .
cd ../..
```

### Option 2: Download Checkpoints

```bash
# Create directory
mkdir -p models/pretrained

# Download SAM2 checkpoint
cd models/pretrained
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt -O sam2_vit_b.pt
cd ../..
```

Or use the make command:
```bash
make download-sam2-checkpoints
```

## Verification

Check if SAM2 is working:

```bash
conda activate noon2
python << EOF
from src.models.sam2_segmentation import SAM2Segmentor
seg = SAM2Segmentor(device="cpu")
print("Using placeholder:" if seg.use_placeholder else "Using real SAM2")
EOF
```

## Benefits of Placeholder

1. **No Installation Failures**: System always works
2. **Quick Testing**: Test without downloading large checkpoints
3. **Development**: Develop other features without SAM2
4. **Fallback**: Graceful degradation if SAM2 fails

## Limitations of Placeholder

- ⚠️ Less accurate segmentation
- ⚠️ Simple threshold-based detection
- ⚠️ No learned features
- ⚠️ Best for testing/development only

## Production Recommendation

For production use:
1. Install SAM2 properly: `make install-sam2`
2. Download checkpoints: `make download-sam2-checkpoints`
3. Verify: Check logs don't show "Using placeholder model"

## Error Messages

### Using Placeholder
```
WARNING  | Using placeholder model for development
INFO     | Placeholder model loaded on mps
```

### Using Real SAM2
```
INFO     | SAM2 model loaded on mps
```

## Files Modified

- `src/models/sam2_segmentation.py` - Enhanced placeholder implementation
- Added `PlaceholderSAM2` with required attributes
- Added `PlaceholderPredictor` class
- Added conditional predictor selection

## Status

✅ **FIXED** - System now works with or without SAM2 installation!
