# ML Microservice Dockerfile
# This container runs the ML inference API as a standalone microservice

FROM python:3.10-slim

# Install system dependencies for OpenCV and other ML libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire ML source code
COPY . .

# Install SAM2 if needed (based on your Makefile)
RUN if [ -f Makefile ]; then \
    apt-get update && apt-get install -y make && \
    make install-sam2 || echo "SAM2 installation skipped"; \
    fi

# Create necessary directories
RUN mkdir -p models data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose the ML API port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/health')"

# Run the ML API server
CMD ["python", "run_api.py"]
