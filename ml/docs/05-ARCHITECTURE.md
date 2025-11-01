# System Architecture

Overview of the Food Recognition ML system architecture.

> 📖 **See also**:
> - [RECOGNITION_ARCHITECTURE.md](../RECOGNITION_ARCHITECTURE.md) - Detailed architecture
> - [docs/PROJECT_STRUCTURE.md](../docs/PROJECT_STRUCTURE.md) - Project structure

---

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Food Recognition System                │
└─────────────────────────────────────────────────────────┘
                            │
           ┌────────────────┼────────────────┐
           ▼                ▼                ▼
    ┌───────────┐    ┌───────────┐    ┌──────────┐
    │  Training │    │   Testing │    │   API    │
    └───────────┘    └───────────┘    └──────────┘
```

---

## Core Components

### 1. Data Processing

```python
src/data_process/
├── classification_dataset.py   # PyTorch Dataset
├── food_labels.py             # Label management
└── preprocess.py              # Data preprocessing
```

**Purpose**: Load and preprocess food images for training

---

### 2. Models

```python
src/models/
├── food_recognition.py         # Main recognition model
└── __init__.py                # Model exports
```

**Architecture**:
- **Backbone**: EfficientNet-B0/B3/B4, ResNet-50
- **Head**: Classification layer (101 or 18 or 115 classes)
- **Optional**: Nutrition regression head

---

### 3. Training

```python
src/training/
├── classification_trainer.py   # Training loop
├── classification_metrics.py   # Metrics calculation
├── mixup.py                   # Data augmentation
└── lr_scheduler.py            # Learning rate scheduling
```

**Features**:
- Auto-resume from crashes
- Mixup/CutMix augmentation
- Cosine annealing scheduler
- Automatic checkpointing

---

### 4. Evaluation

```python
src/evaluation/
└── test_basic_metrics.py      # Model testing
```

**Metrics**: Accuracy, top-5 accuracy, precision, recall, F1

---

## Model Architecture

### FoodRecognitionModel

```python
class FoodRecognitionModel(nn.Module):
    def __init__(self, num_classes=101, backbone='efficientnet_b3'):
        self.backbone = create_backbone(backbone)  # Feature extractor
        self.classifier = nn.Linear(features, num_classes)  # Classifier

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
```

### Supported Backbones

| Backbone | Parameters | Speed | Accuracy |
|----------|------------|-------|----------|
| EfficientNet-B0 | 4.8M | Fast | ~78% |
| EfficientNet-B3 | 12M | Medium | ~91% |
| EfficientNet-B4 | 19M | Slow | ~93% |
| ResNet-50 | 25M | Medium | ~88% |

---

## Training Pipeline

```
1. Load Data
   ├─ Read parquet files (train/val)
   ├─ Apply augmentation (train only)
   └─ Create batches

2. Training Loop
   ├─ Forward pass
   ├─ Calculate loss
   ├─ Backward pass
   ├─ Update weights
   └─ Log metrics

3. Validation
   ├─ Evaluate on val set
   ├─ Calculate metrics
   ├─ Save checkpoints
   └─ Update learning rate

4. Checkpointing
   ├─ Save best accuracy
   ├─ Save best F1
   └─ Save last checkpoint (for resume)
```

---

## Data Flow

```
Raw Images (Food-101)
        │
        ▼
Preprocessing (preprocess_data.py)
        │
        ▼
Parquet Files (train/val/test.parquet)
        │
        ▼
PyTorch Dataset (classification_dataset.py)
        │
        ▼
DataLoader (with augmentation)
        │
        ▼
Model Training (classification_trainer.py)
        │
        ▼
Trained Model (.pt file)
        │
        ▼
Evaluation (test_basic_metrics.py)
        │
        ▼
Results (JSON + visualizations)
```

---

## Configuration

```python
config.py
├─ data_root        # Data directory path
├─ models_root      # Models directory path
├─ device           # cuda/mps/cpu
├─ batch_size       # Training batch size
├─ learning_rate    # Initial learning rate
├─ image_size       # Input image size
└─ mixed_precision  # Enable AMP
```

---

## File Structure

```
ml/
├── src/                    # Source code
│   ├── config.py          # Configuration
│   ├── data_process/      # Data processing
│   ├── models/            # Model definitions
│   ├── training/          # Training infrastructure
│   └── evaluation/        # Testing scripts
│
├── data/                  # Data directory
│   ├── raw/              # Raw datasets
│   └── processed/        # Preprocessed data
│
├── models/               # Trained models
│   └── recognition/      # Recognition models
│
├── scripts/              # Utility scripts
│   └── test_model_quality.sh
│
└── docs/                 # Documentation
    ├── README.md
    ├── 01-SETUP.md
    ├── 02-TRAINING.md
    └── ...
```

---

## Key Design Decisions

### 1. Parquet for Data Storage

**Why**: Fast loading, columnar format, compression

```python
# Instead of loading images individually
df = pd.read_parquet('train.parquet')
# Contains: image_path, food_class, dataset, ...
```

### 2. Separate Label Manager

**Why**: Support multiple datasets with different label spaces

```python
label_manager = FoodLabelManager('food-101')  # 101 classes
label_manager = FoodLabelManager('nutrition5k')  # 18 classes
label_manager = FoodLabelManager('combined')  # 115 classes
```

### 3. Auto-Resume Functionality

**Why**: Training crashes (file descriptors, memory) - don't waste hours

```python
# Automatically checks for last_checkpoint.pt
# Resumes from last epoch
```

### 4. Platform-Specific Optimizations

**Why**: macOS has different constraints than Linux/Windows

```python
if platform.system() == 'Darwin':  # macOS
    num_workers = 0  # Prevent file descriptor leak
    torch.multiprocessing.set_sharing_strategy('file_system')
```

---

## Performance Optimizations

### 1. Memory Management

```python
# Aggressive cleanup on MPS
if device == "mps":
    del tensors
    torch.mps.empty_cache()
    gc.collect()
```

### 2. Data Loading

```python
# No persistent workers (prevents file leaks)
persistent_workers=False

# Pin memory for faster GPU transfer
pin_memory=True if device in ["cuda", "mps"] else False
```

### 3. Mixed Precision (CUDA only)

```python
with autocast():
    logits = model(images)
    loss = criterion(logits, labels)
```

---

## Extension Points

### Adding New Datasets

1. Create preprocessor in `src/data_process/`
2. Add to `FoodLabelManager`
3. Update `preprocess_data.py`

### Adding New Models

1. Implement in `src/models/`
2. Register in `models/__init__.py`
3. Add to training script choices

### Adding New Metrics

1. Implement in `src/training/classification_metrics.py`
2. Update trainer to track metric
3. Add to test scripts

---

## Additional Resources

- **[RECOGNITION_ARCHITECTURE.md](../RECOGNITION_ARCHITECTURE.md)** - Detailed architecture
- **[docs/PROJECT_STRUCTURE.md](../docs/PROJECT_STRUCTURE.md)** - File organization

---

**System architecture complete!** For implementation details, see source code documentation.
