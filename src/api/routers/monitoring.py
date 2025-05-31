"""Monitoring endpoints for the API"""
import os
from datetime import datetime

import psutil
from fastapi import APIRouter

from src.api.dependencies import get_prediction_service

router = APIRouter(prefix="/monitor", tags=["monitoring"])


@router.get("/health/detailed")
async def detailed_health():
    """Detailed health check with system metrics"""
    try:
        # Get prediction service to check model is loaded
        service = get_prediction_service()
        model_loaded = service.model is not None

        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "model": {
                "loaded": model_loaded,
                "type": service.model_info.get("model_type", "unknown"),
                "version": service.model_info.get("model_version", "unknown"),
            },
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available / 1024 / 1024,
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / 1024 / 1024 / 1024,
            },
            "environment": {
                "railway_environment": os.getenv("RAILWAY_ENVIRONMENT", "unknown"),
                "region": os.getenv("RAILWAY_REGION", "unknown"),
                "deployment_id": os.getenv("RAILWAY_DEPLOYMENT_ID", "unknown"),
            },
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/metrics")
async def metrics():
    """Prometheus-style metrics endpoint"""
    # Get metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()

    # Format as Prometheus metrics
    metrics_text = f"""# HELP api_cpu_usage_percent CPU usage percentage
# TYPE api_cpu_usage_percent gauge
api_cpu_usage_percent {cpu_percent}

# HELP api_memory_usage_percent Memory usage percentage
# TYPE api_memory_usage_percent gauge
api_memory_usage_percent {memory.percent}

# HELP api_memory_available_bytes Available memory in bytes
# TYPE api_memory_available_bytes gauge
api_memory_available_bytes {memory.available}
"""

    return metrics_text
