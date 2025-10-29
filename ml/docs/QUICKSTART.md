# Quick Start Guide

Get up and running with the Food Detection & Nutrition Analysis system in **under 10 minutes** using conda and the automated Makefile!

## Prerequisites

- **Conda** (Anaconda or Miniconda)
- macOS with Apple Silicon (M3/M4) or any system with Python 3.11
- 16GB RAM recommended
- 10GB free disk space
- Git installed

## Installation (2 minutes) ⚡

```bash
# 1. Navigate to project directory
cd ml

# 2. Run complete installation (creates conda env + installs deps + SAM2)
make install

# 3. Activate conda environment
conda activate noon2

# 4. Verify installation
python scripts/verify_installation.py
```

That's it! The Makefile handles everything automatically including creating the `noon2` conda environment.

## Initial Setup (First time only)

### Complete Quickstart (One Command!)

```bash
# This runs: install → setup → preprocess
make quickstart
```

This single command:
- ✅ Installs all dependencies including SAM2
- ✅ Creates .env configuration
- ✅ Initializes USDA database
- ✅ Preprocesses all datasets

### Or Step-by-Step

```bash
# 1. Initialize environment and database
make setup

# 2. Preprocess datasets (10-30 minutes)
make preprocess

# 3. View statistics
make preprocess-stats
```

## Basic Usage

### Option 1: Run Inference (No Training Required)

**Using Make (Recommended)**:
```bash
# Run demo with sample data
make demo

# Or analyze your own image
make inference IMAGE=path/to/food.jpg LABELS="rice,chicken,broccoli"

# Detection only (no nutrition)
make inference-detect IMAGE=path/to/food.jpg
```

**Manual**:
```bash
python scripts/inference.py \
    --image path/to/food.jpg \
    --labels "rice,chicken,broccoli" \
    --save-viz \
    --output results/
```

### Option 2: Train Model

**Using Make (Recommended)**:
```bash
# Quick training (10 epochs for testing)
make train-quick

# Full training (50 epochs)
make train

# Custom parameters
make train EPOCHS=100 BATCH_SIZE=16 DEVICE=mps
```

**Manual**:
```bash
python scripts/train.py --epochs 50 --batch-size 8 --device mps
```

### Option 3: Start API Server

**Using Make (Recommended)**:
```bash
# Start server
make serve

# Development mode with auto-reload
make serve-dev

# Test API
make test-api
```

**Manual**:
```bash
python scripts/run_server.py --host 0.0.0.0 --port 8000

# Test it
curl http://localhost:8000/health
```

## Common Tasks

### Analyze a Food Image

```bash
# Using Make
make inference IMAGE=food.jpg LABELS="rice,chicken,broccoli"

# Manual
python scripts/inference.py \
    --image food.jpg \
    --labels "rice,chicken,broccoli" \
    --save-viz \
    --output results/
```

### Check System Status

```bash
# View complete system status
make status

# View project info
make info

# See all available commands
make help
```

### Search USDA Database

```bash
# Activate environment first
source venv/bin/activate

# Then run Python
python << EOF
from src.services import USDALookupService

service = USDALookupService()
results = service.search("chicken breast", limit=5)

for food in results:
    print(f"{food['description']}: {food['energy_kcal']} kcal/100g")
EOF
```

### API Request Example

```bash
# Start server
make serve-bg

# Upload and analyze an image
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@food_image.jpg" \
  -F "food_labels=rice,chicken,vegetables" \
  | jq .

# Stop server
make serve-stop
```

## Multi-Node Training (Advanced)

If you have 2 Macs on the same network:

**Using Make (Recommended)**:

Mac 1 (Master - 192.168.1.100):
```bash
make train-master NUM_NODES=2 MASTER_ADDR=192.168.1.100
```

Mac 2 (Worker - 192.168.1.101):
```bash
make train-worker NUM_NODES=2 NODE_RANK=1 MASTER_ADDR=192.168.1.100
```

**Manual**:

Mac 1 (Master):
```bash
python scripts/train.py \
    --num-nodes 2 --node-rank 0 \
    --master-addr 192.168.1.100 --master-port 29500 \
    --epochs 50 --device mps
```

Mac 2 (Worker):
```bash
python scripts/train.py \
    --num-nodes 2 --node-rank 1 \
    --master-addr 192.168.1.100 --master-port 29500 \
    --epochs 50 --device mps
```

## Troubleshooting

### Installation Issues

```bash
# Re-run installation
make clean
make install

# Verify
python scripts/verify_installation.py
```

### "No module named 'src'"

Make sure you're in the `ml/` directory with activated environment:
```bash
cd ml
source venv/bin/activate
```

### "SAM2 not available"

```bash
# Install SAM2 separately
make install-sam2

# Or the system will use placeholder model (works but less accurate)
```

### Out of Memory

```bash
# Using Make
make train BATCH_SIZE=2

# Or manual
python scripts/train.py --batch-size 2
```

### MPS Not Available

```bash
# Fall back to CPU
make train DEVICE=cpu
```

### Want to Start Fresh?

```bash
# Clean generated files
make clean

# Clean processed data (keeps raw)
make clean-data

# Clean trained models
make clean-models

# Remove virtual environment (nuclear option)
make purge
```

## Next Steps

1. **Improve Model**: Train with more epochs and data
2. **Fine-tune**: Adjust hyperparameters in `src/config.py`
3. **Add Features**: Extend the codebase for your use case
4. **Deploy**: Set up the API server on a production server

## Project Structure Quick Reference

```
ml/
├── scripts/           # Run these!
│   ├── preprocess_data.py
│   ├── train.py
│   ├── inference.py
│   └── run_server.py
├── src/              # Source code (don't modify unless extending)
├── data/             # Your data goes here
├── models/           # Trained models saved here
└── requirements.txt  # Dependencies
```

## Performance Tips

- **Training**: Use MPS backend on Apple Silicon (2-3x faster)
- **Inference**: Reduce image size for faster processing
- **API**: Use async endpoints for concurrent requests
- **Database**: USDA lookups are cached after first query

## Getting Help

Check the main README.md for:
- Detailed API documentation
- Configuration options
- Architecture overview
- Development guidelines

---

**Ready to detect food and analyze nutrition!** 🍔🥗🍕
