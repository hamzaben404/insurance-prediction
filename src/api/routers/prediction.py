# src/api/routers/prediction.py
from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_prediction_service
from src.api.models.insurance import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.api.services.prediction_service import PredictionService

router = APIRouter(
    prefix="/predictions",
    tags=["predictions"],
    responses={404: {"description": "Not found"}},
)


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    prediction_service: PredictionService = Depends(get_prediction_service),
):
    """
    Make a single prediction
    """
    try:
        # Convert request to dict and make prediction
        features = [request.model_dump()]
        results = prediction_service.predict(features)

        return PredictionResponse(**results[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    prediction_service: PredictionService = Depends(get_prediction_service),
):
    """
    Make batch predictions
    """
    try:
        # Convert requests to list of dicts
        features = [record.model_dump() for record in request.records]
        results = prediction_service.predict(features)

        # Format response
        predictions = [PredictionResponse(**result) for result in results]
        return BatchPredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
