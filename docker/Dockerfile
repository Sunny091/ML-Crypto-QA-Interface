# Dockerfile for Crypto Price Prediction ML System
# 支援 Flask API 部署

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY configs/ ./configs/
COPY data/ ./data/
COPY etl/ ./etl/
COPY models/ ./models/
COPY api/ ./api/
COPY evaluation/ ./evaluation/
COPY .env.example ./.env

# Create directories for data and models
RUN mkdir -p /app/data/raw /app/data/processed /app/models/artifacts

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default command: run Flask API
CMD ["python", "-m", "api.app"]
