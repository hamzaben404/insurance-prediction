# tests/integration/test_pipeline.py
from unittest.mock import MagicMock, patch

import numpy as np


def test_data_to_prediction_pipeline(sample_data, mock_model_path):
    """Test the full data-to-prediction pipeline"""
    # Import necessary components
    from src.api.services.prediction_service import PredictionService
    from src.data.preprocess import preprocess_data

    # Process the data
    processed_data = preprocess_data(sample_data)
    # featured_data = create_feature_pipeline(processed_data)

    # Convert to list of records for prediction
    data_records = processed_data.iloc[0:1].to_dict("records")

    # Instead of mocking _ensure_features_match, mock the actual model
    with patch("src.api.services.prediction_service.PredictionService._load_model") as mock_load:
        # Create a mock model with predict_proba method
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

        # Make the _load_model method return our mock
        mock_load.return_value = mock_model

        # Create prediction service
        service = PredictionService(mock_model_path)

        # Make prediction
        predictions = service.predict(data_records)

        # Verify prediction structure
        assert isinstance(predictions, list)
        assert len(predictions) == 1
        assert "prediction" in predictions[0]
        assert "probability" in predictions[0]
        assert isinstance(predictions[0]["prediction"], int)
        assert isinstance(predictions[0]["probability"], float)
