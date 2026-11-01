# Use official lightweight Python 3.12 image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python API dependencies
COPY requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-api.txt

# Copy required application artifacts and source code
COPY api/ ./api/
COPY models/ ./models/
COPY data/processed/dataset_metadata.json ./data/processed/
COPY outputs/model_metrics.csv ./outputs/

# Create logs directory for production inference tracking
RUN mkdir -p logs/inference

# Expose API port
EXPOSE 8000

# Health check to monitor container health
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch FastAPI prediction service
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
