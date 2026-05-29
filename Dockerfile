# PEREGRINE — Aerospace Anomaly Detection System
# CapsNet edge-cloud deployment container

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY best_capsnet_dior.pth ./best_capsnet_dior.pth 2>/dev/null || true

# Set Python path
ENV PYTHONPATH=/app/src

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run PEREGRINE API
CMD ["python", "src/api.py"]
