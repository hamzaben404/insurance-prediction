# src/api/routers/prediction.py
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_prediction_service
from src.api.models.insurance import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    InsuranceFeatures,
    InsurancePrediction,
    ModelInfo,
)
from src.api.services.prediction_service import PredictionService

router = APIRouter(
    prefix="/predictions",
    tags=["predictions"],
    responses={404: {"description": "Not found"}},
)


@router.post("/predict", response_model=InsurancePrediction)
async def predict(
    features: InsuranceFeatures,
    prediction_service: PredictionService = Depends(get_prediction_service),
):
    """Make a single prediction"""
    try:
        # Convert Pydantic model to dict
        features_dict = features.dict()

        # Make prediction
        predictions = prediction_service.predict([features_dict])

        return predictions[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=BatchPredictionResponse)
async def batch_predict(
    request: BatchPredictionRequest,
    prediction_service: PredictionService = Depends(get_prediction_service),
):
    """Make batch predictions"""
    try:
        # Convert list of Pydantic models to list of dicts
        features_dicts = [features.dict() for features in request.records]

        # Make predictions
        predictions = prediction_service.predict(features_dicts)

        return BatchPredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
