# src/api/routers/prediction.py
import json
import logging
import os
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_prediction_service
from src.api.models.insurance import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.api.services.prediction_service import PredictionService

# --- Prediction Logger Setup ---
# This setup creates a dedicated logger for recording predictions.
# It ensures the 'logs' directory exists and writes each prediction as a new line
# in a JSON Lines file, which is excellent for structured logging.

LOGS_DIR = "logs"
PREDICTION_LOG_FILE = os.path.join(LOGS_DIR, "prediction_log.jsonl")

# Create the directory if it doesn't exist
os.makedirs(LOGS_DIR, exist_ok=True)

# Set up a specific logger for prediction events
prediction_logger = logging.getLogger("prediction_logger")
prediction_logger.setLevel(logging.INFO)

# Avoid adding the same handler multiple times if the module is reloaded
if not prediction_logger.handlers:
    file_handler = logging.FileHandler(PREDICTION_LOG_FILE)
    # The formatter ensures that only the raw message (our JSON string) is logged
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    prediction_logger.addHandler(file_handler)
# --- End Logger Setup ---

# Standard logger for general application messages (like request received/finished)
logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["Predictions"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"},
    },
)


@router.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(
    request: PredictionRequest,
    prediction_service: PredictionService = Depends(get_prediction_service),
):
    """
    Make a single prediction for insurance purchase propensity.
    """
    logger.info("Received single prediction request.")
    try:
        features = [request.model_dump()]
        results = prediction_service.predict(features=features)

        if not results:
            logger.error("Prediction service returned no results for single prediction.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prediction service failed to return a result.",
            )

        prediction_result = results[0]

        # --- Log the Prediction Record ---
        try:
            log_record = {
                "prediction_id": str(uuid.uuid4()),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model_version": prediction_service.get_info().get("model_version", "unknown"),
                "input_features": request.model_dump(),
                "prediction_output": prediction_result,
            }
            # Convert the dictionary to a JSON string and write it to the log file
            prediction_logger.info(json.dumps(log_record))
        except Exception as log_e:
            # Important: If logging fails, we log an error to the main logger
            # but we do NOT fail the API request. The user still gets their prediction.
            logger.error(f"Failed to write prediction to log: {log_e}", exc_info=True)
        # --- End Logging ---

        logger.info("Single prediction successful.")
        return PredictionResponse(**prediction_result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during single prediction: {e}", exc_info=True)
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
    """
    logger.info(f"Received batch prediction request with {len(request.records)} records.")
    if not request.records:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch prediction request cannot be empty.",
        )

    try:
        features = [record.model_dump() for record in request.records]
        results = prediction_service.predict(features=features)

        if len(results) != len(features):
            logger.error(
                f"Mismatch in results from service. Expected {len(features)}, got {len(results)}"
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prediction service returned an inconsistent number of results.",
            )

        # --- Log the Batch of Predictions ---
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            model_version = prediction_service.get_info().get("model_version", "unknown")
            for i, record in enumerate(request.records):
                log_record = {
                    "prediction_id": str(uuid.uuid4()),
                    "timestamp_utc": timestamp,
                    "model_version": model_version,
                    "input_features": record.model_dump(),
                    "prediction_output": results[i],
                }
                prediction_logger.info(json.dumps(log_record))
        except Exception as log_e:
            logger.error(f"Failed to write batch predictions to log: {log_e}", exc_info=True)
        # --- End Logging ---

        predictions_response_list = [PredictionResponse(**result) for result in results]
        logger.info(f"Batch prediction successful for {len(predictions_response_list)} records.")
        return BatchPredictionResponse(predictions=predictions_response_list)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during batch processing: {str(e)}",
        )
