# Setup & Installation Guide

Complete guide for installing and setting up the Food Recognition ML system.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Install](#quick-install-recommended)
3. [Manual Installation](#manual-installation)
4. [Verification](#verification)
5. [Data Setup](#data-setup)
6. [Environment Configuration](#environment-configuration)
7. [Troubleshooting](#troubleshooting-setup-issues)

---

## Prerequisites

### System Requirements

- **Operating System**: macOS (Apple Silicon recommended), Linux, or Windows with WSL
- **Memory**: 16GB+ RAM recommended
- **Storage**: 50GB+ free space for datasets and models
- **Python**: 3.10 or 3.11
- **Conda**: Anaconda or Miniconda

### For macOS (Apple Silicon)

- **Device**: M1, M2, M3, or M4 chip
- **macOS**: 12.0 (Monterey) or later
- **MPS Support**: For GPU acceleration

### For NVIDIA GPUs

- **CUDA**: 11.8 or later
- **cuDNN**: Compatible version
- **NVIDIA Driver**: Latest recommended

---

## Quick Install (Recommended)

### Step 1: Clone Repository

```bash
cd /path/to/your/projects
git clone <repository-url>
cd ml
```

### Step 2: Create Conda Environment

```bash
# Create environment with Python 3.11
conda create -n noon2 python=3.11 -y

# Activate environment
conda activate noon2
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

### Step 4: Setup Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings (optional)
nano .env
```

### Step 5: Verify Installation

```bash
# Run verification script
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS Available: {torch.backends.mps.is_available()}')"
```

**Expected output** (on macOS with Apple Silicon):
```
PyTorch: 2.1.0
MPS Available: True
```

---

## Manual Installation

### 1. Install Conda

If you don't have Conda installed:

**macOS/Linux**:
```bash
# Download Miniforge (recommended for Apple Silicon)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh

# Run installer
bash Miniforge3-MacOSX-arm64.sh

# Follow prompts and restart terminal
```

**Verify Conda**:
```bash
conda --version
# Should show: conda 23.x.x or later
```

### 2. Create Python Environment

```bash
# Create environment
conda create -n noon2 python=3.11 numpy pandas scikit-learn -y

# Activate
conda activate noon2

# Verify
python --version
# Should show: Python 3.11.x
```

### 3. Install PyTorch

**For macOS (Apple Silicon)**:
```bash
pip install torch torchvision torchaudio
```

**For NVIDIA GPUs**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Verify PyTorch**:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available() if torch.cuda.is_built() else torch.backends.mps.is_available())"
```

### 4. Install ML Libraries

```bash
# Core ML libraries
pip install \
    timm \
    albumentations \
    opencv-python \
    pillow \
    scikit-learn \
    scipy

# Data processing
pip install \
    pandas \
    pyarrow \
    tqdm

# Logging and visualization
pip install \
    loguru \
    matplotlib \
    seaborn

# Utilities
pip install \
    python-dotenv \
    pyyaml
```

### 5. Install Optional Dependencies

**For API server** (if needed):
```bash
pip install fastapi uvicorn python-multipart
```

**For SAM2 segmentation** (if needed):
```bash
cd .tmp
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd ../..
```

---

## Verification

### Run Verification Tests

```bash
# Check Python version
python --version

# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check device availability
python -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'MPS Available: {torch.backends.mps.is_available()}')
print(f'CPU Available: True')
"

# Check key libraries
python -c "
import timm
import albumentations
import cv2
import pandas
import loguru
print('All libraries imported successfully!')
"
```

### Test Model Loading

```bash
# Test if models can be loaded
python -c "
from src.models import FoodRecognitionModel
model = FoodRecognitionModel(num_classes=101)
print(f'Model created successfully!')
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
"
```

---

## Data Setup

### Directory Structure

Create the required directories:

```bash
# Create data directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models/recognition
mkdir -p results
mkdir -p visualizations
```

### Download Datasets

#### Food-101 Dataset

1. **Download**:
   ```bash
   cd data/raw
   wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
   tar -xzf food-101.tar.gz
   ```

2. **Verify**:
   ```bash
   ls data/raw/food-101/
   # Should show: images/ meta/
   ```

#### Nutrition5k Dataset (Optional)

1. **Download from**: https://github.com/google-research-datasets/Nutrition5k
2. **Place in**: `data/raw/nutrition5k/`

### Preprocess Data

```bash
# Preprocess datasets
python src/train/preprocess_data.py --dataset food-101

# Expected output:
# ✓ Loaded 101,000 images from Food-101
# ✓ Created train.parquet (70,700 samples)
# ✓ Created val.parquet (15,150 samples)
# ✓ Created test.parquet (15,150 samples)
```

**Verify processed data**:
```bash
ls data/processed/
# Should show: train.parquet  val.parquet  test.parquet
```

---

## Environment Configuration

### Create .env File

```bash
# Copy template
cp .env.example .env
```

### Configure Settings

Edit `.env`:

```bash
# Data paths
FOOD_DATA_ROOT=./data
FOOD_MODELS_ROOT=./models

# Training
FOOD_BATCH_SIZE=16
FOOD_LEARNING_RATE=0.001
FOOD_EPOCHS=150
FOOD_DEVICE=mps  # or cuda or cpu

# Model
FOOD_IMAGE_SIZE=224

# Logging
LOG_LEVEL=INFO
```

### Verify Configuration

```bash
# Test configuration loading
python -c "
from config import config
print(f'Data root: {config.data_root}')
print(f'Device: {config.device}')
print(f'Image size: {config.image_size}')
"
```

---

## Troubleshooting Setup Issues

### Issue: Conda environment conflicts

```bash
# Remove old environment
conda deactivate
conda env remove -n noon2

# Create fresh environment
conda create -n noon2 python=3.11 -y
conda activate noon2
pip install -r requirements.txt
```

### Issue: PyTorch not detecting MPS

```bash
# Check macOS version
sw_vers

# MPS requires macOS 12.3+
# Update macOS if needed

# Reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio
```

### Issue: Out of memory during setup

```bash
# Install packages one at a time
pip install torch
pip install torchvision
pip install albumentations
# ... etc
```

### Issue: Permission errors

```bash
# Use --user flag
pip install --user -r requirements.txt

# Or fix permissions
sudo chown -R $USER:$USER /path/to/project
```

### Issue: "Module not found" errors

```bash
# Ensure you're in correct directory
pwd
# Should be: /path/to/noon2/ml

# Ensure conda environment is activated
conda activate noon2

# Reinstall requirements
pip install -r requirements.txt
```

### Issue: File descriptor limit (macOS)

```bash
# Increase file descriptor limit
ulimit -n 4096

# Make permanent (add to ~/.zshrc or ~/.bashrc)
echo "ulimit -n 4096" >> ~/.zshrc
source ~/.zshrc
```

---

## Next Steps

Once installation is complete:

1. **✅ Setup Complete!** - You're ready to start training

2. **Read Training Guide**: See [02-TRAINING.md](02-TRAINING.md) for training instructions

3. **Quick Start Training**:
   ```bash
   python src/train/train_recognition.py \
       --dataset food-101 \
       --dev-mode \
       --epochs 2 \
       --device mps
   ```

4. **For Production Training**: See [02-TRAINING.md](02-TRAINING.md) for full training procedures

---

## Additional Resources

- **Training Guide**: [02-TRAINING.md](02-TRAINING.md)
- **Testing Guide**: [03-TESTING.md](03-TESTING.md)
- **Architecture**: [05-ARCHITECTURE.md](05-ARCHITECTURE.md)
- **Troubleshooting**: [04-TROUBLESHOOTING.md](04-TROUBLESHOOTING.md)

---

**Installation complete!** 🎉 Ready to train food recognition models.
