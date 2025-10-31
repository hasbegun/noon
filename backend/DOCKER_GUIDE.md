# Docker Guide for Backend API

This guide explains how to run the Backend API in a Docker container with access to Ollama running on the host machine.

## Prerequisites

- Docker installed on your machine
- Docker Compose installed (optional, but recommended)
- Ollama running on your host machine (port 11434)
- ML models trained and available in the `../ml` directory

## Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Start the service:**
   ```bash
   make compose-up-build
   ```
   Or:
   ```bash
   docker-compose up -d --build
   ```

2. **View logs:**
   ```bash
   make compose-logs
   ```

3. **Stop the service:**
   ```bash
   make compose-down
   ```

### Option 2: Using Docker directly

1. **Build the image:**
   ```bash
   make docker-build
   ```

2. **Run the container:**
   ```bash
   make docker-run-detached
   ```

3. **View logs:**
   ```bash
   make docker-logs
   ```

4. **Stop the container:**
   ```bash
   make docker-stop
   ```

## Available Make Commands

### Local Development (without Docker)
- `make install` - Install dependencies in virtual environment
- `make start` - Start the API server locally
- `make stop` - Stop the local server
- `make clean` - Clean up virtual environment

### Docker Commands
- `make docker-build` - Build the Docker image
- `make docker-build-no-cache` - Build without using cache
- `make docker-run` - Run container in foreground
- `make docker-run-detached` - Run container in background
- `make docker-run-dev` - Run container with bash shell
- `make docker-stop` - Stop the running container
- `make docker-logs` - Show container logs
- `make docker-exec` - Open bash shell in running container
- `make docker-clean` - Remove container and image

### Docker Compose Commands
- `make compose-up` - Start services in background
- `make compose-up-build` - Build and start services
- `make compose-down` - Stop and remove containers
- `make compose-down-volumes` - Stop and remove volumes
- `make compose-logs` - Show service logs
- `make compose-restart` - Restart services
- `make compose-ps` - Show service status
- `make compose-build` - Build services

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and customize as needed:

```bash
cp .env.example .env
```

Key environment variables:

- `OLLAMA_API_URL` - URL for Ollama API (default: http://host.docker.internal:11434/api/generate)
- `LLAMACPP_API_URL` - URL for Llama.cpp API (default: http://host.docker.internal:8088/v1/chat/completions)
- `BLOOMFILTER_SIZE` - Bloom filter size (default: 100)
- `BLOOMFILTER_FPR` - Bloom filter false positive rate (default: 0.000001)

Note: ML inference is now integrated directly into the backend. The ML models are loaded from the mounted `../ml` directory.

### Accessing Host Services from Container

The container uses `host.docker.internal` to access services running on the host machine. This is automatically configured using:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

This allows the container to connect to:
- Ollama on `host.docker.internal:11434`
- ML Service on `host.docker.internal:8001`
- Llama.cpp on `host.docker.internal:8088`

## Development Workflow

### Hot Reload

The container mounts the `./app` directory, so any changes to your Python code will automatically reload the server:

```bash
make compose-up-build
# Edit files in ./app/
# Changes are reflected immediately
```

### Debugging

To debug inside the container:

```bash
make docker-run-dev
# Inside container:
root@container:/app# python -c "from config import settings; print(settings.OLLAMA_API_URL)"
root@container:/app# curl http://host.docker.internal:11434/api/tags
```

### Testing Host Connectivity

To verify the container can reach Ollama on the host:

```bash
make docker-exec
# Inside container:
curl http://host.docker.internal:11434/api/tags
```

## Architecture

```
┌──────────────────────────────────────────────────┐
│              Docker Container                    │
│  ┌───────────────────────────────────────────┐  │
│  │   FastAPI Backend (port 8000)             │  │
│  │   ┌─────────────────────────────────┐     │  │
│  │   │  Integrated ML Inference        │     │  │
│  │   │  (SAM2 + Volume Estimation)     │     │  │
│  │   └─────────────────────────────────┘     │  │
│  └───────────────┬───────────────────────────┘  │
│                  │                               │
│  Mounted Volumes:│                               │
│  - ../ml (ML code & models)                      │
│  - ./app (backend code)                          │
└──────────────────┼───────────────────────────────┘
                   │
              ┌────▼─────┐
              │  Ollama  │
              │   :11434 │
              └──────────┘
          (Running on host)
```

## Troubleshooting

### Container can't reach Ollama

1. Verify Ollama is running on host:
   ```bash
   curl http://localhost:11434/api/tags
   ```

2. Check if `host.docker.internal` resolves inside container:
   ```bash
   make docker-exec
   ping host.docker.internal
   ```

3. On Linux, you might need to use `--network host` or configure differently

### Permission Issues

If you encounter permission issues with mounted volumes:

```bash
# Check user ID in container
make docker-exec
id

# Match with host user if needed
docker run --user $(id -u):$(id -g) ...
```

### Port Already in Use

If port 8000 is already in use:

1. Change the port in `docker-compose.yml`:
   ```yaml
   ports:
     - "8001:8000"  # Host:Container
   ```

2. Or stop the conflicting service:
   ```bash
   lsof -ti:8000 | xargs kill -9
   ```

## Health Check

The container includes a health check that runs every 30 seconds:

```bash
# Check health status
docker ps
# Look for "healthy" status

# Or use compose
make compose-ps
```

## Production Deployment

For production:

1. Remove `--reload` from CMD in Dockerfile
2. Set `DEBUG=false` in environment
3. Use proper secrets management
4. Set up reverse proxy (nginx/traefik)
5. Configure logging and monitoring
6. Use production-grade database

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [FastAPI Docker Documentation](https://fastapi.tiangolo.com/deployment/docker/)
