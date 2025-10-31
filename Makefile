# Root Makefile for Noon2 Microservices
# Manages all three services: Ollama, ML Inference, and Backend API

.PHONY: help up down logs status health clean restart pull-models

# --- Default Target ---
help: ## Show this help message
	@echo "Noon2 Microservices - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Quick Start:"
	@echo "  1. Start Ollama on host: ollama serve"
	@echo "  2. make up              # Start ML & Backend services"
	@echo "  3. make health          # Check all services"
	@echo "  4. Visit http://localhost:8000/docs"

# --- Service Management ---
up: ## Start ML and Backend services (Ollama runs on host)
	@echo ">>> Starting ML and Backend services..."
	@echo ">>> Note: Make sure Ollama is running on host (ollama serve)"
	@docker-compose up -d
	@echo ">>> Services are starting. This may take a few minutes..."
	@echo ">>> Check status with: make status"
	@echo ">>> Check health with: make health"

up-build: ## Build and start ML and Backend services
	@echo ">>> Building and starting ML and Backend services..."
	@echo ">>> Note: Make sure Ollama is running on host (ollama serve)"
	@docker-compose up -d --build

down: ## Stop all services
	@echo ">>> Stopping all services..."
	@docker-compose down

down-volumes: ## Stop all services and remove volumes
	@echo ">>> Stopping all services and removing volumes..."
	@docker-compose down -v

restart: ## Restart all services
	@echo ">>> Restarting all services..."
	@docker-compose restart

restart-backend: ## Restart backend API only
	@docker-compose restart backend-api

restart-ml: ## Restart ML service only
	@docker-compose restart ml-service

restart-ollama: ## Restart Ollama (on host, not in Docker)
	@echo ">>> Ollama runs on host, not in Docker"
	@echo ">>> To restart Ollama:"
	@echo "    1. pkill ollama"
	@echo "    2. ollama serve"

# --- Logs ---
logs: ## Show logs from all services
	@docker-compose logs -f

logs-backend: ## Show backend API logs only
	@docker-compose logs -f backend-api

logs-ml: ## Show ML service logs only
	@docker-compose logs -f ml-service

logs-ollama: ## Show Ollama logs (runs on host)
	@echo ">>> Ollama runs on host, not in Docker"
	@echo ">>> Check host Ollama logs with:"
	@echo "    journalctl -u ollama -f  (systemd)"
	@echo "    or check console output if running 'ollama serve'"

# --- Status & Health ---
status: ## Show status of all services
	@docker-compose ps

health: ## Check health of all services
	@echo ">>> Checking all services health..."
	@echo ""
	@echo "Ollama Service (Host):"
	@curl -f http://localhost:11434/api/tags > /dev/null 2>&1 && \
		echo "  ✓ Ollama is healthy (http://localhost:11434)" || \
		echo "  ✗ Ollama is not responding - Start with: ollama serve"
	@echo ""
	@echo "ML Service (Docker):"
	@curl -f http://localhost:8001/health > /dev/null 2>&1 && \
		echo "  ✓ ML service is healthy (http://localhost:8001)" || \
		echo "  ✗ ML service is not responding"
	@echo ""
	@echo "Backend API (Docker):"
	@curl -f http://localhost:8000/health > /dev/null 2>&1 && \
		echo "  ✓ Backend API is healthy (http://localhost:8000)" || \
		echo "  ✗ Backend API is not responding"
	@echo ""
	@echo ">>> Service Endpoints:"
	@echo "  - Backend API:  http://localhost:8000/docs"
	@echo "  - ML Service:   http://localhost:8001/docs"
	@echo "  - Ollama (Host): http://localhost:11434"

# --- Ollama Management (Host) ---
pull-models: ## Pull required Ollama models (on host)
	@echo ">>> Pulling Ollama models on host (llama2)..."
	@ollama pull llama2
	@echo ">>> Additional models you might want:"
	@echo "  - ollama pull llama3"
	@echo "  - ollama pull mistral"
	@echo "  - ollama pull phi"

pull-model: ## Pull a specific Ollama model (use: make pull-model MODEL=llama3)
	@echo ">>> Pulling Ollama model on host: $(MODEL)..."
	@ollama pull $(MODEL)

list-models: ## List installed Ollama models (on host)
	@ollama list

# --- Development ---
shell-backend: ## Open shell in backend container
	@docker exec -it noon2-backend-api /bin/bash

shell-ml: ## Open shell in ML service container
	@docker exec -it noon2-ml-service /bin/bash

shell-ollama: ## Interact with Ollama (runs on host)
	@echo ">>> Ollama runs on host, not in Docker"
	@echo ">>> Use: ollama run llama2"

# --- Cleanup ---
clean: ## Stop services and clean up containers/images
	@echo ">>> Cleaning up..."
	@docker-compose down -v
	@docker system prune -f
	@echo ">>> Cleanup complete"

# --- Testing ---
test-backend: ## Test backend API endpoint
	@echo ">>> Testing backend API..."
	@curl http://localhost:8000/health | jq .

test-ml: ## Test ML service endpoint
	@echo ">>> Testing ML service..."
	@curl http://localhost:8001/health | jq .

test-ollama: ## Test Ollama endpoint (on host)
	@echo ">>> Testing Ollama on host..."
	@curl http://localhost:11434/api/tags | jq .

# --- Build ---
build: ## Build all services without starting
	@docker-compose build

build-backend: ## Build backend service only
	@docker-compose build backend-api

build-ml: ## Build ML service only
	@docker-compose build ml-service
