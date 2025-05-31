# src/api/routers/health.py
"""Health check endpoints for the API"""

import logging  # Import logging
from datetime import datetime

from fastapi import APIRouter, Depends

# Assuming get_prediction_service is correctly located and importable
from src.api.dependencies import get_prediction_service
from src.api.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)  # Define a logger for this module

router = APIRouter(
    # prefix="/health", # Consider adding a prefix here if not handled in main.py, but you handle paths fully below
    tags=["Health Checks"],
    responses={404: {"description": "Not found"}},
)


@router.get(
    "/health",  # Full path if no prefix on router include
    summary="Basic API Health Check",
    description="Returns the current operational status and timestamp of the API.",
)
@router.head("/health")
async def health_check():
    """
    Performs a basic health check of the API.
    Returns a status indicating if the API is operational.
    """
    logger.debug("Basic health check requested.")
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@router.get(
    "/health/model",  # Full path if no prefix on router include
    summary="Model Health Check",
    description="Checks if the prediction model is loaded and provides model information.",
)
@router.head("/health/model")
async def model_health(service: PredictionService = Depends(get_prediction_service)):
    """
    Checks the health of the machine learning model.
    Indicates if the model is loaded and provides basic information about it.
    """
    logger.debug("Model health check requested.")
    try:
        # Assuming PredictionService has these methods:
        model_loaded = service.is_model_loaded()
        model_info = service.get_model_info() if model_loaded else None

        if not model_loaded:
            logger.warning("Model health check: Model is not loaded.")

        return {
            "status": "healthy" if model_loaded else "unhealthy",
            "model_loaded": model_loaded,
            "model_info": model_info,
        }
    except AttributeError as e:
        logger.error(
            f"Model health check: PredictionService is missing a required method: {e}",
            exc_info=True,
        )
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": f"PredictionService interface error: {str(e)}",
        }
    except Exception as e:
        logger.error(f"Error during model health check: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": f"An error occurred while checking model health: {str(e)}",
        }


@router.get(
    "/simple_status",  # Full path if no prefix on router include
    status_code=200,
    summary="Simple Liveness Check",
    description="A very lightweight endpoint for uptime monitoring services like UptimeRobot.",
)
@router.head("/simple_status", status_code=200)
async def simple_status_check():
    """
    Provides a minimal '200 OK' response if the API is running.
    Ideal for frequent polling by external monitoring services.
    """
    logger.debug("Simple status check requested (for UptimeRobot).")
    return {"status": "ok_i_am_healthy"}


# Test endpoint for Sentry - make sure to remove or disable this in true production
@router.get(
    "/sentry-debug",
    summary="Sentry Debug Endpoint",
    description="Raises a ZeroDivisionError to test Sentry error reporting. REMOVE IN PRODUCTION.",
    include_in_schema=False,  # Hides it from OpenAPI docs
)
async def trigger_sentry_error():  # Renamed for clarity
    """
    Intentionally raises a ZeroDivisionError to test Sentry integration.
    This endpoint should be removed or disabled in a production environment.
    """
    logger.info("Sentry debug endpoint triggered. Attempting to divide by zero.")
    try:
        1 / 0  # CORRECTED: Directly perform the operation that causes the error
        return {"message": "This will not be reached"}  # Should not happen
    except ZeroDivisionError as e:
        logger.error("ZeroDivisionError successfully triggered for Sentry debug.", exc_info=True)
        # Sentry should capture this automatically.
        # Re-raise if you want FastAPI's default 500 handler to also process it,
        # or let Sentry handle it and return a custom message if preferred.
        # For testing, re-raising ensures it's treated as an unhandled server error.
        raise e
