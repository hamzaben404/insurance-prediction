# src/api/dependencies.py
import os
from functools import lru_cache

from src.api.services.prediction_service import PredictionService
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


@lru_cache()
def get_prediction_service() -> PredictionService:
    """Get prediction service instance (cached)"""
    # Get model path from environment variable or use default
    model_path = os.getenv("MODEL_PATH", "models/comparison/production/production_model.pkl")

    try:
        return PredictionService(model_path)
    except Exception as e:
        logger.error(f"Failed to initialize prediction service: {e}")
        # Return service anyway - it will use dummy model
        return PredictionService(model_path)
