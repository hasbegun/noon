# Makefile for managing the FastAPI Image Analysis Service

# --- Variables ---
# Use python3 as the default interpreter
PYTHON = python3
# Define the virtual environment directory
VENV_DIR = venv
# The path to the virtual environment's activate script
VENV_ACTIVATE = $(VENV_DIR)/bin/activate
# The main application module path for Uvicorn
APP_MODULE = app.main:app
# Docker image name
IMAGE_NAME = innox/noon-api-server
IMAGE_TAG = latest

# --- Phony Targets ---
# These are targets that don't represent files.
.PHONY: help all install start stop clean docker-build docker-run docker-run-dev docker-stop docker-clean compose-up compose-down compose-logs compose-restart health check-ollama check-ml-service check-all-services pull-ollama-models services-up services-down services-logs

# --- Default Target ---
# The default command that runs when you just type "make".
# This lists all available commands.
help:
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# --- Project Lifecycle Commands ---
all: install start ## Install dependencies and start the server

venv: ## Create the Python virtual environment if it doesn't exist
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo ">>> Creating virtual environment..."; \
		$(PYTHON) -m venv $(VENV_DIR); \
	else \
		echo ">>> Virtual environment already exists."; \
	fi

install: venv ## Install project dependencies from requirements.txt
	@echo ">>> Installing dependencies..."
	@. $(VENV_ACTIVATE) && pip install -r requirements.txt
	@. $(VENV_ACTIVATE) && CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python

install-dev: venv
	@echo ">>> Installing dev dependencies..."
	@. $(VENV_ACTIVATE) && pip install -r requirements-dev.txt
	@. $(VENV_ACTIVATE) && CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python


start: ## Start the FastAPI server with auto-reload
	@echo ">>> Starting server at http://127.0.0.1:8000..."
	@. $(VENV_ACTIVATE) && uvicorn $(APP_MODULE) --reload

stop: ## Stop any running instances of the server
	@echo ">>> Stopping server..."
	@pkill -f "$(APP_MODULE)" || echo "Server was not running."

clean: ## Remove the virtual environment and __pycache__ directories
	@echo ">>> Cleaning up project..."
	@rm -rf $(VENV_DIR)
	@find . -type d -name "__pycache__" -exec rm -r {} +
	@echo "Cleanup complete."

# --- Docker Commands ---
docker-build: ## Build the Docker image
	@echo ">>> Building Docker image..."
	@docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

docker-build-no-cache: ## Build the Docker image without cache
	@echo ">>> Building Docker image (no cache)..."
	@docker build --no-cache -t $(IMAGE_NAME):$(IMAGE_TAG) .

docker-run: ## Run the container with volume mount and host access
	@echo ">>> Running Docker container..."
	@docker run --rm \
		--name noon-api-server \
		-v ./app:/app \
		-v ../ml:/ml:ro \
		-v ./data:/data \
		-p 8000:8000 \
		--add-host host.docker.internal:host-gateway \
		-e OLLAMA_API_URL=http://host.docker.internal:11434/api/generate \
		-e LLAMACPP_API_URL=http://host.docker.internal:8088/v1/chat/completions \
		$(IMAGE_NAME):$(IMAGE_TAG)

docker-run-detached: ## Run the container in detached mode
	@echo ">>> Running Docker container in background..."
	@docker run -d \
		--name noon-api-server \
		-v ./app:/app \
		-v ../ml:/ml:ro \
		-v ./data:/data \
		-p 8000:8000 \
		--add-host host.docker.internal:host-gateway \
		-e OLLAMA_API_URL=http://host.docker.internal:11434/api/generate \
		-e LLAMACPP_API_URL=http://host.docker.internal:8088/v1/chat/completions \
		$(IMAGE_NAME):$(IMAGE_TAG)
	@echo ">>> Container started. View logs with: make docker-logs"

docker-run-dev: ## Run the container with bash shell for development
	@echo ">>> Running Docker container (dev shell)..."
	@docker run -it --rm \
		--name noon-api-server-dev \
		-v ./app:/app \
		-v ../ml:/ml:ro \
		-v ./data:/data \
		-p 8000:8000 \
		--add-host host.docker.internal:host-gateway \
		-e OLLAMA_API_URL=http://host.docker.internal:11434/api/generate \
		-e LLAMACPP_API_URL=http://host.docker.internal:8088/v1/chat/completions \
		$(IMAGE_NAME):$(IMAGE_TAG) /bin/bash

docker-stop: ## Stop the running container
	@echo ">>> Stopping Docker container..."
	@docker stop noon-api-server || echo "Container is not running."

docker-logs: ## Show container logs
	@docker logs -f noon-api-server

docker-exec: ## Execute bash in running container
	@docker exec -it noon-api-server /bin/bash

docker-clean: ## Remove Docker image and stopped containers
	@echo ">>> Cleaning up Docker resources..."
	@docker rm -f noon-api-server noon-api-server-dev 2>/dev/null || true
	@docker rmi $(IMAGE_NAME):$(IMAGE_TAG) 2>/dev/null || true
	@echo ">>> Docker cleanup complete."

# --- Docker Compose Commands (Legacy - using local docker-compose.yml) ---
compose-up: ## Start backend service only using local docker-compose
	@echo ">>> Starting backend service with local docker-compose..."
	@docker-compose up -d
	@echo ">>> Backend service started. View logs with: make compose-logs"

compose-up-build: ## Build and start backend service using local docker-compose
	@echo ">>> Building and starting backend service with local docker-compose..."
	@docker-compose up -d --build

compose-down: ## Stop backend service using local docker-compose
	@echo ">>> Stopping backend service..."
	@docker-compose down

compose-down-volumes: ## Stop backend service and remove volumes
	@echo ">>> Stopping backend service and removing volumes..."
	@docker-compose down -v

compose-logs: ## Show logs from local docker-compose services
	@docker-compose logs -f

compose-restart: ## Restart local docker-compose services
	@echo ">>> Restarting backend service..."
	@docker-compose restart

compose-ps: ## Show status of local docker-compose services
	@docker-compose ps

compose-build: ## Build local docker-compose services
	@echo ">>> Building backend service..."
	@docker-compose build

# --- Microservices Commands (Root docker-compose.yml) ---
services-up: ## Start ML + Backend services (Ollama runs on host)
	@echo ">>> Starting ML Service + Backend API..."
	@echo ">>> Note: Make sure Ollama is running on host (ollama serve)"
	@cd .. && docker-compose up -d
	@echo ">>> Services starting. Check health with: make check-all-services"
	@echo ">>> View logs with: make services-logs"

services-up-build: ## Build and start ML + Backend services
	@echo ">>> Building and starting ML and Backend services..."
	@echo ">>> Note: Make sure Ollama is running on host (ollama serve)"
	@cd .. && docker-compose up -d --build
	@echo ">>> Services starting. Check health with: make check-all-services"

services-down: ## Stop all services
	@echo ">>> Stopping all services..."
	@cd .. && docker-compose down

services-down-volumes: ## Stop all services and remove volumes
	@echo ">>> Stopping all services and removing volumes..."
	@cd .. && docker-compose down -v

services-logs: ## Show logs from all services
	@cd .. && docker-compose logs -f

services-logs-backend: ## Show backend API logs only
	@cd .. && docker-compose logs -f backend-api

services-logs-ml: ## Show ML service logs only
	@cd .. && docker-compose logs -f ml-service

services-logs-ollama: ## Show Ollama logs (runs on host)
	@echo ">>> Ollama runs on host, not in Docker"
	@echo ">>> Check host Ollama logs with:"
	@echo "    journalctl -u ollama -f  (systemd)"
	@echo "    or check console output if running 'ollama serve'"

services-restart: ## Restart all services
	@echo ">>> Restarting all services..."
	@cd .. && docker-compose restart

services-restart-backend: ## Restart backend API only
	@echo ">>> Restarting backend API..."
	@cd .. && docker-compose restart backend-api

services-restart-ml: ## Restart ML service only
	@echo ">>> Restarting ML service..."
	@cd .. && docker-compose restart ml-service

services-ps: ## Show status of all services
	@cd .. && docker-compose ps

# --- Health Check Commands ---
check-ollama: ## Check if Ollama service is healthy (on host)
	@echo ">>> Checking Ollama health (host)..."
	@curl -f http://localhost:11434/api/tags > /dev/null 2>&1 && \
		echo "✓ Ollama is healthy (http://localhost:11434)" || \
		echo "✗ Ollama is not responding - Start with: ollama serve"

check-ml-service: ## Check if ML service is healthy
	@echo ">>> Checking ML service health..."
	@curl -f http://localhost:8001/health > /dev/null 2>&1 && \
		echo "✓ ML service is healthy" || \
		echo "✗ ML service is not responding"

check-backend: ## Check if Backend API is healthy
	@echo ">>> Checking Backend API health..."
	@curl -f http://localhost:8000/health > /dev/null 2>&1 && \
		echo "✓ Backend API is healthy" || \
		echo "✗ Backend API is not responding"

check-all-services: ## Check health of all services
	@echo ">>> Checking all services health..."
	@echo ""
	@make check-ollama
	@make check-ml-service
	@make check-backend
	@echo ""
	@echo ">>> Service endpoints:"
	@echo "  - Backend API:  http://localhost:8000/docs"
	@echo "  - ML Service:   http://localhost:8001/docs"
	@echo "  - Ollama (Host): http://localhost:11434"

health: check-all-services ## Alias for check-all-services

# --- Ollama Management (Host) ---
pull-ollama-models: ## Pull required Ollama models on host
	@echo ">>> Pulling Ollama models on host..."
	@ollama pull llama2
	@echo ">>> Models pulled successfully"
	@echo ">>> Pull additional models with: ollama pull <model>"

ollama-list: ## List installed Ollama models (on host)
	@ollama list

ollama-shell: ## Interact with Ollama (runs on host)
	@echo ">>> Ollama runs on host, not in Docker"
	@echo ">>> Use: ollama run llama2"
