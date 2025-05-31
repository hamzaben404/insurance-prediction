# src/api/routers/health.py
"""Health check endpoints for the API"""

from datetime import datetime

from fastapi import APIRouter, Depends  # Added Depends for potential future use with dependencies

# Assuming get_prediction_service is correctly located and importable
# If src.api.dependencies might not be on PYTHONPATH in some test environments,
# consider relative imports if appropriate or ensure PYTHONPATH is set.
from src.api.dependencies import get_prediction_service
from src.api.services.prediction_service import (  # Explicitly import for type hinting
    PredictionService,
)

# It's good practice to handle potential import errors for dependencies if they are optional
# or might not be available in all environments, though get_prediction_service seems core.


router = APIRouter(
    tags=["Health Checks"],  # Standardized tag name
    responses={404: {"description": "Not found"}},  # Default response for this router
)


@router.get(
    "/health",
    summary="Basic API Health Check",
    description="Returns the current operational status and timestamp of the API.",
)
@router.head("/health")  # Explicitly support HEAD method
async def health_check():
    """
    Performs a basic health check of the API.
    Returns a status indicating if the API is operational.
    """
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@router.get(
    "/health/model",
    summary="Model Health Check",
    description="Checks if the prediction model is loaded and provides model information.",
)
@router.head("/health/model")  # Explicitly support HEAD method
async def model_health(
    service: PredictionService = Depends(get_prediction_service),
):  # Use Depends for service
    """
    Checks the health of the machine learning model.
    Indicates if the model is loaded and provides basic information about it.
    """
    try:
        model_loaded = service.is_model_loaded()  # Assuming PredictionService has such a method
        model_info = service.get_model_info() if model_loaded else None  # Assuming method exists

        return {
            "status": "healthy" if model_loaded else "unhealthy",
            "model_loaded": model_loaded,
            "model_info": model_info,
        }
    except Exception as e:
        # Log the exception here with a logger for better debugging
        # logger.error(f"Error during model health check: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": f"An error occurred while checking model health: {str(e)}",
        }


@router.get(
    "/simple_status",
    status_code=200,
    summary="Simple Liveness Check",
    description="A very lightweight endpoint for uptime monitoring services like UptimeRobot.",
)
@router.head("/simple_status", status_code=200)  # Explicitly support HEAD method
async def simple_status_check():
    """
    Provides a minimal '200 OK' response if the API is running.
    Ideal for frequent polling by external monitoring services.
    """
    # For HEAD requests, FastAPI/Starlette automatically strips the response body.
    # So, returning a body here is fine; it just won't be sent for HEAD.
    return {"status": "ok_i_am_healthy"}


# Example of how your PredictionService might look for the above to work smoothly:
# class PredictionService:
#     def __init__(self):
#         self.model = None # Load your model here
#         self.model_info = {} # Populate with model details
#
#     def load_model(self, model_path: str = "path/to/your/model.pkl"):
#         # Actual model loading logic
#         # self.model = joblib.load(model_path)
#         # self.model_info = {"name": "InsurancePredictor", "version": "1.0"}
#         self.model = "dummy_model" # Placeholder
#         self.model_info = {"type": "LGBMClassifier", "version": "0.1.0_prod"} # Placeholder
#
#     def is_model_loaded(self) -> bool:
#         return self.model is not None
#
#     def get_model_info(self) -> dict:
#         return self.model_info if self.model else {}
#
# # And in dependencies.py
# # from src.api.services.prediction_service import PredictionService
# #
# # prediction_service_instance = PredictionService()
# # prediction_service_instance.load_model() # Load model on startup
# #
# # def get_prediction_service():
# #     return prediction_service_instance
