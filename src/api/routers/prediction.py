# src/api/routers/prediction.py
import logging  # It's good practice to log errors

from fastapi import APIRouter, Depends, HTTPException, status  # Added status for HTTPException

from src.api.dependencies import get_prediction_service
from src.api.models.insurance import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.api.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)  # Define a logger

router = APIRouter(
    # prefix="/predictions", # REMOVED: Prefix is handled in main.py when including this router
    tags=["Predictions"],  # Standardized tag name slightly
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"},  # Good to define common error responses
    },
)


@router.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(
    request: PredictionRequest,
    prediction_service: PredictionService = Depends(get_prediction_service),
):
    """
    Make a single prediction for insurance purchase propensity.

    - **Input**: Features of a single customer.
    - **Output**: Prediction (0 or 1) and probability.
    """
    logger.info(f"Received single prediction request: {request.model_dump(mode='json')}")
    try:
        # Convert request to dict and make prediction
        # The service expects a list of feature dicts
        features = [request.model_dump()]
        results = prediction_service.predict(
            features=features
        )  # Pass as keyword argument for clarity

        if not results:  # Handle case where prediction service returns empty
            logger.error("Prediction service returned no results for single prediction.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prediction service failed to return a result.",
            )

        logger.info(f"Single prediction result: {results[0]}")
        return PredictionResponse(**results[0])
    except HTTPException:  # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        logger.error(f"Error during single prediction: {e}", exc_info=True)
        # Sentry should capture this
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )


@router.post("/batch", response_model=BatchPredictionResponse, status_code=status.HTTP_200_OK)
async def predict_batch(
    request: BatchPredictionRequest,
    prediction_service: PredictionService = Depends(get_prediction_service),
):
    """
    Make batch predictions for insurance purchase propensity.

    - **Input**: A list of customer features.
    - **Output**: A list of predictions and probabilities.
    """
    logger.info(f"Received batch prediction request with {len(request.records)} records.")
    if not request.records:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch prediction request cannot be empty.",
        )

    try:
        # Convert requests to list of dicts
        features = [record.model_dump() for record in request.records]
        results = prediction_service.predict(features=features)  # Pass as keyword argument

        if len(results) != len(features):
            logger.error(
                f"Mismatch in number of results from prediction service. Expected {len(features)}, got {len(results)}"
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prediction service returned an inconsistent number of results.",
            )

        # Format response
        predictions_response_list = [PredictionResponse(**result) for result in results]
        logger.info(f"Batch prediction successful for {len(predictions_response_list)} records.")
        return BatchPredictionResponse(predictions=predictions_response_list)
    except HTTPException:  # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}", exc_info=True)
        # Sentry should capture this
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during batch processing: {str(e)}",
        )
