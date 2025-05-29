# src/api/dependencies.py
import os
from functools import lru_cache

from src.api.services.prediction_service import PredictionService


@lru_cache()
def get_prediction_service() -> PredictionService:
    """Get prediction service instance (cached)"""
    # Get model path from environment variable or use default
    model_path = os.getenv("MODEL_PATH", "models/comparison/production/production_model.pkl")
    return PredictionService(model_path)
