.PHONY: help install install-sam2 setup preprocess train inference serve clean test lint format check-data check-usda init-db

# Conda environment name
CONDA_ENV := noon2
PYTHON := python3.11
CONDA_ACTIVATE := source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(CONDA_ENV)
CONDA_RUN := conda run -n $(CONDA_ENV)

# Configuration
DEVICE ?= mps
EPOCHS ?= 50
BATCH_SIZE ?= 8
HOST ?= 0.0.0.0
PORT ?= 8000

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Food Detection & Nutrition Analysis - Makefile Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Setup Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Configuration:$(NC)"
	@echo "  DEVICE=$(DEVICE) (mps, cuda, cpu)"
	@echo "  EPOCHS=$(EPOCHS)"
	@echo "  BATCH_SIZE=$(BATCH_SIZE)"
	@echo "  HOST=$(HOST)"
	@echo "  PORT=$(PORT)"
	@echo ""
	@echo "$(GREEN)Examples:$(NC)"
	@echo "  make install         # Complete installation"
	@echo "  make setup          # Initialize everything"
	@echo "  make train          # Train model"
	@echo "  make serve          # Start API server"

# ============================================================================
# INSTALLATION
# ============================================================================

install: ## Install all dependencies (main + SAM2)
	@echo "$(BLUE)Installing Food Detection System with conda...$(NC)"
	@$(MAKE) check-conda
	@$(MAKE) create-conda-env
	@$(MAKE) install-deps
	@$(MAKE) install-sam2
	@$(MAKE) setup-env
	@echo "$(GREEN)✓ Installation complete!$(NC)"
	@echo ""
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Activate environment: conda activate $(CONDA_ENV)"
	@echo "  2. Initialize system: make setup"
	@echo "  3. Run preprocessing: make preprocess"

check-conda: ## Check conda installation
	@echo "$(BLUE)Checking conda installation...$(NC)"
	@command -v conda >/dev/null 2>&1 || { echo "$(RED)Error: conda not found. Please install Anaconda or Miniconda$(NC)"; exit 1; }
	@echo "$(GREEN)✓ Conda found$(NC)"

create-conda-env: ## Create conda environment
	@echo "$(BLUE)Creating conda environment: $(CONDA_ENV)...$(NC)"
	@if conda env list | grep -q "^$(CONDA_ENV) "; then \
		echo "$(YELLOW)Conda environment '$(CONDA_ENV)' already exists$(NC)"; \
	else \
		conda create -n $(CONDA_ENV) python=3.11 -y; \
		echo "$(GREEN)✓ Conda environment created$(NC)"; \
	fi

install-deps: ## Install Python dependencies
	@echo "$(BLUE)Installing Python dependencies...$(NC)"
	@$(CONDA_RUN) pip install --upgrade pip setuptools wheel
	@$(CONDA_RUN) pip install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

install-sam2: ## Install SAM2 from GitHub
	@echo "$(BLUE)Installing SAM2...$(NC)"
	@if $(CONDA_RUN) python -c "import sam2" 2>/dev/null; then \
		echo "$(YELLOW)SAM2 already installed$(NC)"; \
	else \
		mkdir -p .tmp && \
		cd .tmp && \
		git clone https://github.com/facebookresearch/sam2.git && \
		cd sam2 && \
		$(CONDA_RUN) pip install -e . && \
		cd ../.. && \
		echo "$(GREEN)✓ SAM2 installed from GitHub$(NC)"; \
	fi

install-dev: install ## Install with development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	@$(CONDA_RUN) pip install -e .
	@echo "$(GREEN)✓ Development installation complete$(NC)"

# ============================================================================
# SETUP
# ============================================================================

setup: setup-env check-data init-db ## Complete setup (environment, data, database)
	@echo "$(GREEN)✓ Setup complete!$(NC)"

setup-env: ## Create .env file from template
	@echo "$(BLUE)Setting up environment configuration...$(NC)"
	@if [ ! -f ".env" ]; then \
		cp .env.example .env; \
		echo "$(GREEN)✓ Created .env file$(NC)"; \
		echo "$(YELLOW)Please review and adjust .env settings$(NC)"; \
	else \
		echo "$(YELLOW).env already exists$(NC)"; \
	fi

check-data: ## Check data directories
	@echo "$(BLUE)Checking data directories...$(NC)"
	@if [ -d "data/raw" ]; then \
		echo "$(GREEN)✓ data/raw exists$(NC)"; \
		ls -la data/raw/ | tail -n +4 | head -n 10; \
	else \
		echo "$(RED)✗ data/raw not found$(NC)"; \
		echo "$(YELLOW)Please ensure data directory is properly linked$(NC)"; \
	fi
	@if [ -d "data/usda" ]; then \
		echo "$(GREEN)✓ data/usda exists$(NC)"; \
	else \
		echo "$(RED)✗ data/usda not found$(NC)"; \
	fi

check-usda: ## Check USDA data files
	@echo "$(BLUE)Checking USDA nutrition data...$(NC)"
	@if [ -f "data/usda/FoodData_Central_foundation_food_json_2025-04-24.json" ]; then \
		echo "$(GREEN)✓ Foundation food data found$(NC)"; \
	else \
		echo "$(RED)✗ Foundation food data not found$(NC)"; \
	fi
	@if [ -f "data/usda/FoodData_Central_branded_food_json_2025-04-24.json" ]; then \
		echo "$(GREEN)✓ Branded food data found$(NC)"; \
	else \
		echo "$(RED)✗ Branded food data not found$(NC)"; \
	fi

init-db: ## Initialize USDA nutrition database
	@echo "$(BLUE)Initializing USDA nutrition database...$(NC)"
	@$(CONDA_RUN) python -c "\
from src.services import USDALookupService; \
service = USDALookupService(); \
print('$(GREEN)✓ USDA database initialized$(NC)')"

# ============================================================================
# DATA PROCESSING
# ============================================================================

preprocess: ## Preprocess all raw datasets
	@echo "$(BLUE)Preprocessing datasets...$(NC)"
	@$(CONDA_RUN) python scripts/preprocess_data.py
	@echo "$(GREEN)✓ Preprocessing complete$(NC)"

preprocess-stats: ## Show preprocessing statistics
	@echo "$(BLUE)Dataset statistics:$(NC)"
	@if [ -f "data/processed/statistics.json" ]; then \
		$(CONDA_RUN) python -c "\
import json; \
with open('data/processed/statistics.json') as f: \
	stats = json.load(f); \
	print(json.dumps(stats, indent=2))"; \
	else \
		echo "$(RED)No statistics found. Run 'make preprocess' first$(NC)"; \
	fi

# ============================================================================
# TRAINING
# ============================================================================

train: ## Train model (single node)
	@echo "$(BLUE)Training model...$(NC)"
	@$(CONDA_RUN) python scripts/train.py \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--device $(DEVICE)

train-quick: ## Quick training (10 epochs for testing)
	@echo "$(BLUE)Quick training (10 epochs)...$(NC)"
	@$(CONDA_RUN) python scripts/train.py \
		--epochs 10 \
		--batch-size 4 \
		--device $(DEVICE)

train-multi: ## Multi-node training (requires manual setup on each node)
	@echo "$(BLUE)Multi-node training setup:$(NC)"
	@echo ""
	@echo "$(YELLOW)On Master Node (192.168.1.100):$(NC)"
	@echo "  make train-master MASTER_ADDR=192.168.1.100"
	@echo ""
	@echo "$(YELLOW)On Worker Node (192.168.1.101):$(NC)"
	@echo "  make train-worker MASTER_ADDR=192.168.1.100 NODE_RANK=1"

train-master: ## Train as master node
	@echo "$(BLUE)Starting training on master node...$(NC)"
	@$(CONDA_RUN) python scripts/train.py \
		--num-nodes $(NUM_NODES) \
		--node-rank 0 \
		--master-addr $(MASTER_ADDR) \
		--master-port 29500 \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--device $(DEVICE)

train-worker: ## Train as worker node
	@echo "$(BLUE)Starting training on worker node $(NODE_RANK)...$(NC)"
	@$(CONDA_RUN) python scripts/train.py \
		--num-nodes $(NUM_NODES) \
		--node-rank $(NODE_RANK) \
		--master-addr $(MASTER_ADDR) \
		--master-port 29500 \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--device $(DEVICE)

train-resume: ## Resume training from checkpoint
	@echo "$(BLUE)Resuming training...$(NC)"
	@$(CONDA_RUN) python scripts/train.py \
		--checkpoint $(CHECKPOINT) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--device $(DEVICE)

# ============================================================================
# INFERENCE
# ============================================================================

inference: ## Run inference on an image
	@if [ -z "$(IMAGE)" ]; then \
		echo "$(RED)Error: IMAGE not specified$(NC)"; \
		echo "$(YELLOW)Usage: make inference IMAGE=path/to/image.jpg$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Running inference on $(IMAGE)...$(NC)"
	@$(CONDA_RUN) python scripts/inference.py \
		--image $(IMAGE) \
		--labels "$(LABELS)" \
		--save-viz \
		--output results/ \
		--device $(DEVICE)

inference-detect: ## Run detection only (no nutrition)
	@if [ -z "$(IMAGE)" ]; then \
		echo "$(RED)Error: IMAGE not specified$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Running food detection...$(NC)"
	@$(CONDA_RUN) python scripts/inference.py \
		--image $(IMAGE) \
		--detect-only \
		--save-viz \
		--output results/ \
		--device $(DEVICE)

demo: ## Run demo inference on sample image
	@echo "$(BLUE)Running demo inference...$(NC)"
	@if [ -f "data/raw/nutrition5k/imagery/realsense_overhead/1/rgb_00000.png" ]; then \
		$(MAKE) inference-detect IMAGE=data/raw/nutrition5k/imagery/realsense_overhead/1/rgb_00000.png; \
	else \
		echo "$(YELLOW)No sample image found. Please specify IMAGE=path/to/image.jpg$(NC)"; \
	fi

# ============================================================================
# API SERVER
# ============================================================================

serve: ## Start FastAPI server
	@echo "$(BLUE)Starting API server on $(HOST):$(PORT)...$(NC)"
	@$(CONDA_RUN) python scripts/run_server.py \
		--host $(HOST) \
		--port $(PORT)

serve-dev: ## Start server with auto-reload
	@echo "$(BLUE)Starting API server in development mode...$(NC)"
	@$(CONDA_RUN) python scripts/run_server.py \
		--host $(HOST) \
		--port $(PORT) \
		--reload

serve-bg: ## Start server in background
	@echo "$(BLUE)Starting API server in background...$(NC)"
	@mkdir -p logs
	@$(CONDA_RUN) python scripts/run_server.py \
		--host $(HOST) \
		--port $(PORT) > logs/server.log 2>&1 & echo $$! > .server.pid
	@echo "$(GREEN)✓ Server started (PID: $$(cat .server.pid))$(NC)"
	@echo "$(YELLOW)Logs: tail -f logs/server.log$(NC)"

serve-stop: ## Stop background server
	@if [ -f .server.pid ]; then \
		kill $$(cat .server.pid) && rm .server.pid; \
		echo "$(GREEN)✓ Server stopped$(NC)"; \
	else \
		echo "$(YELLOW)No server PID found$(NC)"; \
	fi

test-api: ## Test API endpoints
	@echo "$(BLUE)Testing API endpoints...$(NC)"
	@echo "$(YELLOW)Health check:$(NC)"
	@curl -s http://$(HOST):$(PORT)/health | jq . || echo "$(RED)Server not running$(NC)"

# ============================================================================
# DEVELOPMENT
# ============================================================================

test: ## Run tests
	@echo "$(BLUE)Running tests...$(NC)"
	@$(CONDA_RUN) pytest tests/ -v

test-placeholder: ## Test placeholder SAM2 model
	@echo "$(BLUE)Testing placeholder SAM2 model...$(NC)"
	@$(CONDA_RUN) python test_placeholder.py

lint: ## Run linters
	@echo "$(BLUE)Running linters...$(NC)"
	@$(CONDA_RUN) black --check src/ scripts/
	@$(CONDA_RUN) isort --check-only src/ scripts/

format: ## Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	@$(CONDA_RUN) black src/ scripts/
	@$(CONDA_RUN) isort src/ scripts/
	@echo "$(GREEN)✓ Code formatted$(NC)"

clean: ## Clean generated files
	@echo "$(BLUE)Cleaning generated files...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache .coverage htmlcov
	@echo "$(GREEN)✓ Cleaned$(NC)"

clean-data: ## Clean processed data (keeps raw)
	@echo "$(YELLOW)Cleaning processed data...$(NC)"
	@rm -rf data/processed/*.parquet data/processed/*.json
	@echo "$(GREEN)✓ Processed data cleaned$(NC)"

clean-models: ## Clean trained models
	@echo "$(YELLOW)Cleaning trained models...$(NC)"
	@rm -rf models/segmentation/*.pt models/segmentation/*.pth
	@echo "$(GREEN)✓ Models cleaned$(NC)"

clean-all: clean clean-data ## Clean everything except raw data
	@echo "$(GREEN)✓ Deep clean complete$(NC)"

purge: ## Remove conda environment (WARNING: destructive)
	@echo "$(RED)Removing conda environment '$(CONDA_ENV)'...$(NC)"
	@conda env remove -n $(CONDA_ENV) -y || true
	@rm -rf .tmp
	@echo "$(GREEN)✓ Conda environment removed$(NC)"

# ============================================================================
# UTILITIES
# ============================================================================

status: ## Show system status
	@echo "$(BLUE)System Status:$(NC)"
	@echo ""
	@echo "$(YELLOW)Environment:$(NC)"
	@if conda env list | grep -q "^$(CONDA_ENV) "; then echo "  $(GREEN)✓$(NC) Conda environment '$(CONDA_ENV)'"; else echo "  $(RED)✗$(NC) Conda environment"; fi
	@if [ -f ".env" ]; then echo "  $(GREEN)✓$(NC) .env file"; else echo "  $(RED)✗$(NC) .env file"; fi
	@echo ""
	@echo "$(YELLOW)Data:$(NC)"
	@if [ -d "data/raw" ]; then echo "  $(GREEN)✓$(NC) Raw data"; else echo "  $(RED)✗$(NC) Raw data"; fi
	@if [ -d "data/usda" ]; then echo "  $(GREEN)✓$(NC) USDA data"; else echo "  $(RED)✗$(NC) USDA data"; fi
	@if [ -f "data/processed/train.parquet" ]; then echo "  $(GREEN)✓$(NC) Processed data"; else echo "  $(RED)✗$(NC) Processed data"; fi
	@if [ -f "data/usda/nutrition.db" ]; then echo "  $(GREEN)✓$(NC) USDA database"; else echo "  $(RED)✗$(NC) USDA database"; fi
	@echo ""
	@echo "$(YELLOW)Models:$(NC)"
	@if [ -d "models/pretrained" ]; then echo "  $(GREEN)✓$(NC) Pretrained models dir"; else echo "  $(RED)✗$(NC) Pretrained models dir"; fi
	@if [ -d "models/segmentation" ]; then echo "  $(GREEN)✓$(NC) Segmentation models dir"; else echo "  $(RED)✗$(NC) Segmentation models dir"; fi

logs: ## Create logs directory
	@mkdir -p logs
	@echo "$(GREEN)✓ Logs directory created$(NC)"

info: ## Show project information
	@echo "$(BLUE)Food Detection & Nutrition Analysis System$(NC)"
	@echo ""
	@echo "$(YELLOW)Version:$(NC) 1.0.0"
	@echo "$(YELLOW)Python:$(NC) $(PYTHON)"
	@echo "$(YELLOW)Device:$(NC) $(DEVICE)"
	@echo ""
	@echo "$(YELLOW)Project Structure:$(NC)"
	@tree -L 2 -I 'venv|__pycache__|*.pyc|data|models' . 2>/dev/null || ls -la

shell: ## Open Python shell with environment loaded
	@echo "$(BLUE)Opening Python shell...$(NC)"
	@$(CONDA_RUN) python

notebook: ## Start Jupyter notebook
	@echo "$(BLUE)Starting Jupyter notebook...$(NC)"
	@$(CONDA_RUN) jupyter notebook

conda-info: ## Show conda environment info
	@echo "$(BLUE)Conda Environment Information:$(NC)"
	@conda info --envs | grep $(CONDA_ENV) || echo "$(RED)Environment '$(CONDA_ENV)' not found$(NC)"
	@echo ""
	@$(CONDA_RUN) python --version 2>/dev/null || true
	@$(CONDA_RUN) pip list 2>/dev/null | head -20 || true

# ============================================================================
# QUICKSTART WORKFLOWS
# ============================================================================

quickstart: install setup preprocess ## Complete quickstart setup
	@echo ""
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)  Quickstart Complete!$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Activate environment: $(BLUE)source $(VENV)/bin/activate$(NC)"
	@echo "  2. Train model: $(BLUE)make train$(NC)"
	@echo "  3. Run inference: $(BLUE)make inference IMAGE=path/to/image.jpg$(NC)"
	@echo "  4. Start API: $(BLUE)make serve$(NC)"
	@echo ""
	@echo "$(YELLOW)Or try:$(NC)"
	@echo "  $(BLUE)make demo$(NC) - Run demo inference"
	@echo "  $(BLUE)make train-quick$(NC) - Quick training test"
	@echo "  $(BLUE)make help$(NC) - Show all commands"

all: quickstart ## Alias for quickstart

# ============================================================================
# DOWNLOAD SAM2 CHECKPOINTS
# ============================================================================

download-sam2-checkpoints: ## Download SAM2 pretrained checkpoints
	@echo "$(BLUE)Downloading SAM2 checkpoints...$(NC)"
	@mkdir -p models/pretrained
	@echo "$(YELLOW)Downloading SAM2 ViT-B checkpoint (1.2GB)...$(NC)"
	@cd models/pretrained && \
	if [ ! -f sam2_vit_b.pt ]; then \
		curl -L -o sam2_vit_b.pt "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt" || \
		echo "$(RED)Download failed. Please download manually from https://github.com/facebookresearch/sam2$(NC)"; \
	else \
		echo "$(YELLOW)Checkpoint already exists$(NC)"; \
	fi
	@echo "$(GREEN)✓ Checkpoints ready$(NC)"

check-sam2: ## Check if SAM2 is properly installed
	@echo "$(BLUE)Checking SAM2 installation...$(NC)"
	@$(CONDA_RUN) python scripts/check_sam2.py || true

# ============================================================================
# SHORTCUTS
# ============================================================================

i: install ## Shortcut for install
s: serve ## Shortcut for serve
t: train ## Shortcut for train
p: preprocess ## Shortcut for preprocess
c: clean ## Shortcut for clean
h: help ## Shortcut for help
