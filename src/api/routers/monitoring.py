# src/api/routers/monitoring.py

"""Monitoring endpoints for the API"""
import os
from datetime import datetime

import psutil
from fastapi import APIRouter, Response  # <--- IMPORT Response HERE

# Assuming PredictionService is in this path, or adjust as needed
from src.api.dependencies import get_prediction_service

# from src.api.services.prediction_service import PredictionService # If you use type hinting

router = APIRouter(prefix="/monitor", tags=["monitoring"])


@router.get("/health/detailed")
async def detailed_health():
    # ... (your existing detailed_health code) ...
    # Keep this as is, it's fine to return JSON here.
    try:
        # Get prediction service to check model is loaded
        service = get_prediction_service()
        model_loaded = service.model is not None

        # Get system metrics
        cpu_percent = psutil.cpu_percent(
            interval=None
        )  # Changed interval to None for faster response
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "model": {
                "loaded": model_loaded,
                "type": service.model_info.get("model_type", "unknown")
                if service.model_info
                else "unknown",
                "version": service.model_info.get("model_version", "unknown")
                if service.model_info
                else "unknown",
            },
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": round(memory.available / (1024 * 1024), 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024 * 1024 * 1024), 2),
            },
            "environment": {
                "railway_environment": os.getenv("RAILWAY_ENVIRONMENT", "unknown"),
                "region": os.getenv("RAILWAY_REGION", "unknown"),
                "deployment_id": os.getenv("RAILWAY_DEPLOYMENT_ID", "unknown"),
            },
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/metrics")  # Path is /monitor/metrics because of the router prefix
async def metrics():
    # Get metrics
    # Using interval=None for cpu_percent for potentially faster response in an API context.
    # interval=1 blocks for 1 second.
    cpu_percent = psutil.cpu_percent(interval=None)
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
    # Return using fastapi.Response with the correct media_type
    return Response(
        content=metrics_text, media_type="text/plain; charset=utf-8"
    )  # <--- MODIFIED LINE
