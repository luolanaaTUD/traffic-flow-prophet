# Multi-stage build for traffic-flow-prophet FastAPI backend
# Base image: Python 3.11+ slim (matches pyproject requires-python >=3.11)
# Default: CPU-only PyTorch to keep image usable without GPU

# Build stage: install dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install uv for faster dependency resolution (optional, can use pip directly)
RUN pip install --no-cache-dir uv

# Install project with TimesFM extras (default model)
# Using CPU-only torch to keep image lightweight and usable without GPU
COPY . .
RUN uv pip install --system --no-cache \
    -e ".[timesfm]" \
    --index-url https://download.pytorch.org/whl/cpu

# Runtime stage: minimal image with only runtime dependencies
FROM python:3.11-slim

WORKDIR /app

# Install runtime system dependencies (if any needed)
# Currently none required, but keeping structure for future needs
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app/ /app/app/
COPY pyproject.toml /app/

# Create directory for optional data files (not included in image)
RUN mkdir -p /app/data

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run uvicorn server
# Note: QWeather API keys must be provided at runtime via environment variables
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
