# Dockerfile - Fixed with proper Python path
FROM python:3.10-slim AS builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies as root (not using --user)
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies including curl for healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create directories for data and models with proper permissions
RUN mkdir -p data/raw data/processed/test models/comparison/production config && \
    chmod -R 755 data models config scripts

# Set environment variables
ENV PYTHONPATH=/app
ENV API_PORT=8000
ENV MODEL_PATH=/app/models/comparison/production/production_model.pkl

# Add labels for versioning
LABEL org.opencontainers.image.source="https://github.com/yourusername/insurance-prediction"
LABEL org.opencontainers.image.description="Vehicle Insurance Prediction API"
LABEL org.opencontainers.image.licenses="MIT"

# Create a dummy model for testing (this will be replaced with real model in production)
RUN python scripts/create_dummy_model.py

# Create a dummy test.csv file
RUN echo "id,Gender,Age,HasDrivingLicense,RegionID,VehicleAge,PastAccident,AnnualPremium,SalesChannelID,DaysSinceCreated,Result" > data/raw/test.csv && \
    echo "1,Male,30,1,10,1-2 Year,No,1000,26,60,0" >> data/raw/test.csv

# Create a non-root user and give ownership of app directory
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check using curl
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application (ensure uvicorn is in PATH)
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
