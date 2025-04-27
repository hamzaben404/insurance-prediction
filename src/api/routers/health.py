# src/api/routers/health.py
from datetime import datetime

import psutil
from fastapi import APIRouter, Depends

from src.api.dependencies import get_prediction_service
from src.api.services.prediction_service import PredictionService

router = APIRouter(
    prefix="/health",
    tags=["health"],
    responses={404: {"description": "Not found"}},
)


@router.get("")
async def health():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/model")
async def model_health(
    prediction_service: PredictionService = Depends(get_prediction_service),
):
    """
    Model health check
    """
    # Get model info
    model_info = prediction_service.get_info()

    return {
        "status": "loaded",
        "model_type": model_info.get("model_type", "unknown"),
        "model_version": model_info.get("model_version", "unknown"),
    }


@router.get("/metrics")
async def metrics():
    """
    System metrics endpoint
    """
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage("/").percent,
    }
