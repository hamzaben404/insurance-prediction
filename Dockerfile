# Dockerfile - Add healthcheck and LABEL for versioning
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

# Install dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create a non-root user
RUN useradd -m appuser
USER appuser

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV API_PORT=8000
ENV MODEL_PATH=/app/models/comparison/production/production_model.pkl

# Add labels for versioning
LABEL org.opencontainers.image.source="https://github.com/yourusername/insurance-prediction"
LABEL org.opencontainers.image.description="Vehicle Insurance Prediction API"
LABEL org.opencontainers.image.licenses="MIT"

# Define healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]