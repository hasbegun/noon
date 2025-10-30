# Food Detection & Nutrition Analysis System

A modern ML pipeline for detecting food items from images and providing detailed nutrition analysis using SAM2 (Segment Anything Model 2) and USDA FoodData Central.

> 📖 **Documentation**: See [docs/](docs/) for complete guides
>
> 🏗️ **Project Structure**: See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for detailed layout

## Features

- **Food Detection**: Advanced segmentation using SAM2 for accurate food item detection
- **Volume Estimation**: Intelligent portion size estimation from segmentation masks
- **Nutrition Analysis**: Comprehensive nutrition information from USDA FoodData Central
- **Multi-node Training**: Distributed training support for Apple Silicon (M3/M4)
- **FastAPI Backend**: Production-ready REST API for inference
- **Modern Tech Stack**: Python 3.11, PyTorch, SAM2, FastAPI

## Architecture

```
ml/
├── src/                        # Source code
│   ├── config.py              # Configuration management
│   ├── data/                  # Data processing
│   │   ├── preprocessing.py   # Dataset preprocessing
│   │   ├── dataset.py         # PyTorch dataset
│   │   └── loader.py          # Data loaders with multi-node support
│   ├── models/                # ML models
│   │   ├── sam2_segmentation.py   # SAM2 integration
│   │   ├── food_detector.py       # Food detection pipeline
│   │   └── volume_estimator.py    # Volume estimation
│   ├── services/              # Services
│   │   ├── usda_lookup.py     # USDA nutrition database
│   │   └── nutrition_analyzer.py  # Complete analysis pipeline
│   ├── training/              # Training infrastructure
│   │   ├── distributed.py     # Distributed training utilities
│   │   └── trainer.py         # Training logic
│   └── api/                   # FastAPI backend
│       └── main.py            # API endpoints
├── scripts/                   # Executable scripts
│   ├── preprocess_data.py     # Data preprocessing
│   ├── train.py               # Model training
│   ├── inference.py           # Run inference
│   └── run_server.py          # Start API server
├── data/                      # Data directory (symlink)
│   ├── raw/                   # Raw datasets
│   ├── usda/                  # USDA nutrition data
│   └── processed/             # Processed data
├── models/                    # Models directory (symlink)
│   ├── pretrained/            # Pretrained models
│   └── segmentation/          # Trained models
└── requirements.txt           # Python dependencies
```

## Installation

### Prerequisites

- **Conda** (Anaconda or Miniconda)
- macOS with Apple Silicon (M3/M4) recommended
- 16GB+ RAM recommended
- Git (for SAM2 installation)

### Quick Install (Recommended)

Use the Makefile for automated conda setup:

```bash
cd ml

# Complete installation (creates conda env 'noon2' + dependencies + SAM2)
make install

# Activate conda environment
conda activate noon2

# Verify installation
python scripts/verify_installation.py

# Initialize system
make setup

# You're ready!
make help
```

### Manual Installation

If you prefer manual setup with conda:

1. **Create conda environment**:
```bash
cd ml
conda create -n noon2 python=3.11 -y
conda activate noon2
```

2. **Install dependencies**:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. **Install SAM2 from GitHub**:
```bash
mkdir -p .tmp
cd .tmp
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd ../..
```

4. **Setup environment**:
```bash
cp .env.example .env
# Edit .env as needed
```

See [CONDA_SETUP.md](CONDA_SETUP.md) for detailed conda instructions.

## Usage

### Using Make Commands (Recommended)

The Makefile provides convenient commands for all operations:

```bash
# See all available commands
make help

# Complete quickstart setup
make quickstart

# Individual steps
make preprocess          # Preprocess data
make train              # Train model
make inference IMAGE=path/to/image.jpg LABELS="rice,chicken"
make serve              # Start API server
```

### 1. Data Preprocessing

Process all raw datasets and create train/val/test splits:

**Using Make**:
```bash
make preprocess

# View statistics
make preprocess-stats
```

**Manual**:
```bash
python scripts/preprocess_data.py
```

This will:
- Load all datasets from `data/raw/`
- Validate images and annotations
- Remove missing or corrupted data
- Create train/val/test splits (70/15/15)
- Save processed data to `data/processed/`

### 2. Training

#### Single Node Training

**Using Make**:
```bash
# Full training
make train

# Quick test (10 epochs)
make train-quick

# With custom parameters
make train EPOCHS=100 BATCH_SIZE=16 DEVICE=mps
```

**Manual**:
```bash
python scripts/train.py \
    --epochs 50 \
    --batch-size 8 \
    --lr 1e-4 \
    --device mps
```

#### Multi-Node Training (2 M3/M4 Macs)

**Using Make**:

On Master Node (192.168.1.100):
```bash
make train-master NUM_NODES=2 MASTER_ADDR=192.168.1.100
```

On Worker Node (192.168.1.101):
```bash
make train-worker NUM_NODES=2 NODE_RANK=1 MASTER_ADDR=192.168.1.100
```

**Manual**:

On Master Node (192.168.1.100):
```bash
python scripts/train.py \
    --num-nodes 2 \
    --node-rank 0 \
    --master-addr 192.168.1.100 \
    --master-port 29500 \
    --epochs 50 \
    --batch-size 8 \
    --device mps
```

On Worker Node (192.168.1.101):
```bash
python scripts/train.py \
    --num-nodes 2 \
    --node-rank 1 \
    --master-addr 192.168.1.100 \
    --master-port 29500 \
    --epochs 50 \
    --batch-size 8 \
    --device mps
```

Training arguments:
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size per device (default: 8)
- `--lr`: Learning rate (default: 1e-4)
- `--model-type`: SAM2 variant (vit_b, vit_l, vit_h)
- `--checkpoint`: Resume from checkpoint
- `--num-nodes`: Number of nodes for distributed training
- `--device`: Device (cuda, mps, cpu)

### 3. Inference

#### Detection Only

**Using Make**:
```bash
make inference-detect IMAGE=path/to/food_image.jpg

# Or run demo with sample data
make demo
```

**Manual**:
```bash
python scripts/inference.py \
    --image path/to/food_image.jpg \
    --detect-only \
    --save-viz \
    --output results/
```

#### Full Nutrition Analysis

**Using Make**:
```bash
make inference IMAGE=path/to/food_image.jpg LABELS="rice,chicken,salad"
```

**Manual**:
```bash
python scripts/inference.py \
    --image path/to/food_image.jpg \
    --labels "rice,chicken,salad" \
    --save-viz \
    --output results/
```

This will:
- Detect and segment food items
- Estimate volume and mass
- Look up nutrition information from USDA database
- Calculate total nutrition values
- Save results as JSON and visualizations

### 4. API Server

**Using Make**:
```bash
# Start server
make serve

# Development mode with auto-reload
make serve-dev

# Background mode
make serve-bg

# Stop background server
make serve-stop

# Test API
make test-api
```

**Manual**:
```bash
python scripts/run_server.py --host 0.0.0.0 --port 8000

# Or with auto-reload for development
python scripts/run_server.py --reload
```

#### API Endpoints

**Health Check**:
```bash
curl http://localhost:8000/health
```

**Detect Food Items**:
```bash
curl -X POST http://localhost:8000/api/v1/detect \
  -F "file=@food_image.jpg" \
  -F "return_visualization=true"
```

**Analyze Nutrition**:
```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@food_image.jpg" \
  -F "food_labels=rice,chicken,salad"
```

**Search USDA Database**:
```bash
curl -X POST http://localhost:8000/api/v1/usda/search \
  -H "Content-Type: application/json" \
  -d '{"query": "chicken breast", "limit": 10}'
```

**Get Nutrition for Specific Food**:
```bash
curl http://localhost:8000/api/v1/usda/food/171477?portion_g=150
```

## Configuration

Configuration can be set via environment variables or `.env` file:

```bash
# Data paths
FOOD_DATA_ROOT=./data
FOOD_MODELS_ROOT=./models

# Training
FOOD_BATCH_SIZE=8
FOOD_LEARNING_RATE=0.0001
FOOD_EPOCHS=50
FOOD_NUM_NODES=1

# Model
FOOD_SAM2_MODEL_TYPE=vit_b
FOOD_IMAGE_SIZE=1024

# API
FOOD_API_HOST=0.0.0.0
FOOD_API_PORT=8000

# Device
FOOD_DEVICE=mps
FOOD_MIXED_PRECISION=true
```

## Data Format

### Raw Data Structure

The system supports multiple food image datasets:

- **Nutrition5k**: Images with ground-truth nutrition labels
- **Food-101**: 101 food categories with 1000 images each
- **UECFOOD100**: 100 food categories with bounding boxes
- **iFood2019**: Large-scale food image dataset

Place datasets in `data/raw/`:
```
data/raw/
├── nutrition5k/
├── food-101/
├── UECFOOD100/
└── ifood2019/
```

### USDA Nutrition Data

USDA FoodData Central JSON files should be in `data/usda/`:
```
data/usda/
├── FoodData_Central_foundation_food_json_2025-04-24.json
├── FoodData_Central_branded_food_json_2025-04-24.json
└── nutrition.db (auto-generated)
```

## Model Details

### SAM2 Segmentation

The system uses SAM2 (Segment Anything Model 2) for food segmentation:

- **Automatic Segmentation**: Detects all food items in image
- **Prompted Segmentation**: Uses points or boxes as prompts
- **Post-processing**: Filters and ranks detected regions

### Volume Estimation

Volume estimation uses multiple cues:

1. **2D Area**: From segmentation mask
2. **Shape Analysis**: Circularity and compactness
3. **Depth Estimation**: Optional depth map support
4. **Heuristics**: Food-type specific models

Volume is converted to mass using food-specific density values.

### Nutrition Lookup

Nutrition information is retrieved from USDA FoodData Central:

- **Foundation Foods**: Core reference foods
- **Branded Foods**: Commercial food products
- **SR Legacy**: Historical USDA nutrient database

All nutrition values are scaled to detected portion sizes.

## Multi-Node Training

The system supports distributed training across multiple Apple Silicon Macs:

### Requirements

- Network connectivity between nodes
- Same Python environment on all nodes
- Access to shared data (via NFS or replicated locally)

### Setup

1. **Configure Network**: Ensure nodes can communicate
2. **Sync Code**: Same code on all nodes
3. **Start Training**: Launch script on each node with appropriate rank

### Performance

Multi-node training provides:
- Linear speedup (2x with 2 nodes)
- Synchronized gradients
- Larger effective batch size

## Troubleshooting

### SAM2 Installation Issues

The system works with a placeholder model if SAM2 isn't available. Check status:
```bash
make check-sam2
```

**If using placeholder** (works but less accurate):
```
⚠ Using placeholder model
```

**To install real SAM2**:
```bash
# Install SAM2 from GitHub
make install-sam2

# Download checkpoint (1.2GB)
make download-sam2-checkpoints

# Verify
make check-sam2
```

**Manual installation**:
```bash
cd .tmp
git clone https://github.com/facebookresearch/sam2.git
cd sam2
conda run -n noon2 pip install -e .
```

See [SAM2_FIX.md](SAM2_FIX.md) and [CRITICAL_FIX_SUMMARY.md](CRITICAL_FIX_SUMMARY.md) for details.

### MPS Backend Issues

If MPS (Metal Performance Shaders) doesn't work:
```bash
# Using make
make train DEVICE=cpu

# Manual
python scripts/train.py --device cpu
```

### Out of Memory

Reduce batch size:
```bash
# Using make
make train BATCH_SIZE=2

# Manual
python scripts/train.py --batch-size 2
```

### USDA Database Not Found

Initialize the database manually:
```bash
make init-db
```

Or with Python:
```python
from src.services import USDALookupService
service = USDALookupService()  # This will create the database
```

### AttributeError with SAM2

If you see `AttributeError: 'PlaceholderSAM2' object has no attribute 'image_size'`:
```bash
# This is fixed! Update your code:
git pull  # or re-download the files

# Test the fix:
make test-placeholder
```

## Performance Optimization

### Training Speed

- Use MPS backend on Apple Silicon for 2-3x speedup
- Enable mixed precision training (automatic on CUDA)
- Increase batch size based on available memory
- Use multiple data loading workers

### Inference Speed

- Use smaller SAM2 variant (vit_b vs vit_h)
- Reduce image resolution
- Batch multiple images together
- Cache USDA database queries

## Development

### Code Style

The codebase follows:
- PEP 8 style guide
- Type hints throughout
- Comprehensive docstrings
- Modular architecture

### Testing

```bash
# Run tests (if implemented)
pytest tests/
```

### Adding New Datasets

1. Add processor to `src/data/preprocessing.py`
2. Implement dataset-specific loading logic
3. Ensure consistent output format
4. Run preprocessing pipeline

### Extending USDA Database

1. Download additional USDA JSON files
2. Place in `data/usda/`
3. Update `config.py` with new filenames
4. Re-initialize database

## Citation

This project builds upon:

- **SAM2**: Segment Anything Model 2 by Meta AI
- **USDA FoodData Central**: U.S. Department of Agriculture
- **FoodSAM**: Original inspiration (though using updated tools)

## License

This project is for educational and research purposes. Please cite appropriately if used in publications.

## Support

For issues, questions, or contributions, please open an issue on the project repository.

---

**Built with modern ML tools for accurate food detection and nutrition analysis.**
