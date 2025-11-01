# Project Structure

## Overview

Clean, organized structure for the food detection ML engine with proper separation of concerns.

## Directory Layout

```
ml/
├── README.md                   # Main project documentation
├── Makefile                    # Automated commands (40+)
├── requirements.txt            # Python dependencies
├── .env.example               # Environment config template
│
├── docs/                      # All documentation
│   ├── GET_STARTED.md         # Quick 3-step setup
│   ├── QUICKSTART.md          # Detailed guide
│   ├── CONDA_SETUP.md         # Conda environment setup
│   ├── PERFORMANCE_OPTIMIZATIONS.md  # Performance fixes
│   ├── CHECKPOINT_RESUME_GUIDE.md    # Training resume
│   ├── TRAINING_HANG_FIX.md          # Validation hang fix
│   ├── FIX_ZERO_MASKS.md             # Zero masks problem
│   ├── RESUME_TRAINING_SUMMARY.md    # Resume summary
│   ├── CHANGES_SUMMARY.md            # All changes
│   ├── PROJECT_STRUCTURE.md          # This file
│   └── archive/               # Old/historical docs
│       ├── CRITICAL_FIX_SUMMARY.md
│       ├── FIXED_AND_READY.md
│       ├── LATEST_FIXES.md
│       ├── SAM2_FIX.md
│       └── TEST_SAM2_CHECK.md
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   │
│   ├── api/                   # FastAPI server
│   │   ├── __init__.py
│   │   ├── main.py            # API entry point
│   │   ├── routes.py          # API endpoints
│   │   └── schemas.py         # Pydantic models
│   │
│   ├── data_process/          # Data loading & processing
│   │   ├── __init__.py
│   │   ├── dataset.py         # PyTorch Dataset
│   │   ├── loader.py          # DataLoader creation
│   │   └── preprocessing.py   # Data preprocessing
│   │
│   ├── models/                # Model architectures
│   │   ├── __init__.py
│   │   ├── food_detector.py   # Main detector
│   │   ├── sam2_segmentation.py  # SAM2 integration
│   │   └── volume_estimator.py   # Volume estimation
│   │
│   ├── services/              # Business logic services
│   │   ├── __init__.py
│   │   ├── usda_lookup.py     # USDA nutrition database
│   │   └── inference.py       # Inference service
│   │
│   ├── train/                 # Training scripts
│   │   ├── __init__.py
│   │   ├── train.py           # Main training script
│   │   └── preprocess_data.py # Data preprocessing runner
│   │
│   ├── training/              # Training infrastructure
│   │   ├── __init__.py
│   │   ├── trainer.py         # Trainer class
│   │   └── distributed.py     # Multi-node support
│   │
│   └── utils/                 # Utility scripts
│       ├── __init__.py
│       ├── debug_masks.py     # Mask debugging
│       ├── verify_checkpoint.py  # Checkpoint verification
│       └── test_placeholder.py   # SAM2 testing
│
├── tests/                     # Test files
│   ├── test_dataset.py        # Dataset tests
│   ├── test_food1.jpg         # Test images
│   └── test_food2.jpg
│
├── data/                      # Data (symlink to external storage)
│   ├── raw/                   # Raw datasets
│   │   ├── food-101/
│   │   ├── UECFOOD100/
│   │   ├── nutrition5k/
│   │   └── ...
│   └── processed/             # Preprocessed data
│       ├── train.parquet
│       ├── val.parquet
│       └── test.parquet
│
├── models/                    # Trained models (symlink)
│   ├── pretrained/            # Pretrained weights
│   │   └── sam2_hiera_b+.pt
│   └── segmentation/          # Training checkpoints
│       ├── last_checkpoint.pt
│       ├── best_model.pt
│       └── checkpoint_epoch_*.pt
│
├── visualizations/            # Output visualizations
│   └── ...
│
├── logs/                      # Training logs
│   └── ...
│
├── run_api.py                 # API server entry point
├── debug_masks.sh             # Debug wrapper
└── verify_checkpoint.sh       # Checkpoint verify wrapper
```

## Key Directories

### `/src/` - Source Code

All Python source code organized by function:

- **api/** - REST API endpoints
- **data_process/** - Data loading pipeline
- **models/** - Neural network architectures
- **services/** - Business logic (inference, USDA lookup)
- **train/** - Training scripts
- **training/** - Training infrastructure
- **utils/** - Utility/debug scripts

### `/docs/` - Documentation

All documentation in one place:

- Main guides (GET_STARTED, QUICKSTART, etc.)
- Technical docs (PERFORMANCE_OPTIMIZATIONS, etc.)
- `/archive/` - Historical/superseded docs

### `/tests/` - Tests

Test files and test data:

- Unit tests
- Integration tests
- Test images/fixtures

### `/data/` - Data (External)

Usually a symlink to external storage:

- `/raw/` - Original datasets
- `/processed/` - Preprocessed parquet files

### `/models/` - Model Weights (External)

Usually a symlink to external storage:

- `/pretrained/` - SAM2, other pretrained models
- `/segmentation/` - Training checkpoints

## File Organization Principles

### Python Modules

1. **Keep related code together**
   - All data loading in `data_process/`
   - All models in `models/`
   - All training in `training/`

2. **Use `__init__.py` for clean imports**
   ```python
   from src.models import FoodDetector
   from src.data_process import create_data_loaders
   ```

3. **Separate scripts from modules**
   - Runnable scripts: `src/train/train.py`
   - Reusable modules: `src/training/trainer.py`

### Documentation

1. **Active docs in `/docs/`**
   - Current setup guides
   - Technical documentation
   - Troubleshooting

2. **Archive old docs in `/docs/archive/`**
   - Historical fixes
   - Superseded guides
   - Reference material

3. **Keep README.md concise**
   - Link to detailed docs
   - Quick start only

### Scripts

1. **Entry points in root**
   - `run_api.py` - Start server
   - `*.sh` - Convenience wrappers

2. **Utilities in `/src/utils/`**
   - Debugging tools
   - Validation scripts
   - Testing utilities

## Usage Patterns

### Running Scripts

```bash
# From ml directory:

# Training
python src/train/train.py --epochs 50 --batch-size 8

# Preprocessing
python src/train/preprocess_data.py

# API server
python run_api.py

# Utilities (via wrapper)
./debug_masks.sh
./verify_checkpoint.sh

# Utilities (direct)
python src/utils/debug_masks.py
python -m src.utils.verify_checkpoint
```

### Importing Modules

```python
# In any script with proper path setup:
from src.models import FoodDetector
from src.training import Trainer
from src.data_process import create_data_loaders
from src.config import config
```

### Adding New Code

**New model:**
```bash
# Create: src/models/my_new_model.py
# Update: src/models/__init__.py
```

**New utility:**
```bash
# Create: src/utils/my_utility.py
# Update: src/utils/__init__.py
# Optional: Create wrapper script in root
```

**New documentation:**
```bash
# Active doc: docs/MY_GUIDE.md
# Archive old: docs/archive/OLD_DOC.md
```

## Recent Refactoring (2025-10-30)

### Files Moved

**To `/src/utils/`:**
- `debug_masks.py` → `src/utils/debug_masks.py`
- `verify_checkpoint.py` → `src/utils/verify_checkpoint.py`
- `test_placeholder.py` → `src/utils/test_placeholder.py`

**To `/docs/`:**
- `FIX_ZERO_MASKS.md` → `docs/FIX_ZERO_MASKS.md`
- `RESUME_TRAINING_SUMMARY.md` → `docs/RESUME_TRAINING_SUMMARY.md`

**To `/docs/archive/`:**
- `CRITICAL_FIX_SUMMARY.md` → `docs/archive/`
- `FIXED_AND_READY.md` → `docs/archive/`
- `LATEST_FIXES.md` → `docs/archive/`
- `SAM2_FIX.md` → `docs/archive/`
- `TEST_SAM2_CHECK.md` → `docs/archive/`

**Created:**
- `src/utils/__init__.py` - Utils module init
- `debug_masks.sh` - Wrapper for debug utility
- `verify_checkpoint.sh` - Wrapper for checkpoint utility
- `docs/PROJECT_STRUCTURE.md` - This file

### Backward Compatibility

Wrapper scripts maintain backward compatibility:
```bash
# Old: python debug_masks.py
# New: ./debug_masks.sh  (or python src/utils/debug_masks.py)

# Old: python verify_checkpoint.py
# New: ./verify_checkpoint.sh  (or python src/utils/verify_checkpoint.py)
```

## Maintenance

### Adding New Files

1. **Determine category**: data, model, training, util, doc
2. **Place in appropriate directory**
3. **Update `__init__.py` if needed**
4. **Add to this doc if significant**

### Cleaning Up

```bash
# Remove Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
find . -name "*.pyc" -delete

# Remove old checkpoints
make clean-checkpoints

# Archive old documentation
mv docs/OLD_DOC.md docs/archive/
```

### Checking Structure

```bash
# View tree
tree -L 3 -I '__pycache__|*.pyc'

# List Python modules
find src -name "*.py" | grep -v __pycache__

# List documentation
ls -la docs/*.md
ls -la docs/archive/*.md
```

## Benefits of Current Structure

✅ **Clear organization** - Easy to find files
✅ **Scalable** - Room for growth
✅ **Maintainable** - Logical grouping
✅ **Professional** - Industry standard layout
✅ **Clean imports** - No path hacks needed
✅ **Documented** - This file explains everything

## Quick Reference

| Task | Location |
|------|----------|
| Train model | `src/train/train.py` |
| Run API | `run_api.py` |
| Add new model | `src/models/` |
| Debug data | `./debug_masks.sh` |
| Read docs | `docs/` |
| Tests | `tests/` |
| Config | `src/config.py` |
| Utilities | `src/utils/` |

---

**Last Updated**: 2025-10-30
**Status**: ✅ Refactored and organized
