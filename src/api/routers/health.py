"""Health check endpoints"""
from datetime import datetime

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
@router.head("/health")  # Add HEAD method support
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@router.get("/health/model")
async def model_health():
    """Check model health status"""
    # Check if model is loaded
    try:
        from src.api.dependencies import get_prediction_service

        service = get_prediction_service()
        model_loaded = service.model is not None

        return {
            "status": "healthy" if model_loaded else "unhealthy",
            "model_loaded": model_loaded,
            "model_info": service.model_info if model_loaded else None,
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
