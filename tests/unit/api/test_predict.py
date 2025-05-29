# tests/unit/api/test_predict.py
from unittest.mock import patch


def test_predict_endpoint(test_client, prediction_payload, mock_model_path):
    """Test prediction endpoint with mocked model"""
    # Mock the prediction service
    with patch("src.api.services.prediction_service.PredictionService.predict") as mock_predict:
        # Set up the mock to return a fixed prediction
        mock_predict.return_value = [{"prediction": 1, "probability": 0.75}]

        # Make request
        response = test_client.post("/predictions/predict", json=prediction_payload)

        # Check response
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert data["prediction"] == 1
        assert data["probability"] == 0.75


def test_batch_predict_endpoint(test_client, batch_prediction_payload, mock_model_path):
    """Test batch prediction endpoint with mocked model"""
    # Mock the prediction service
    with patch("src.api.services.prediction_service.PredictionService.predict") as mock_predict:
        # Set up the mock to return fixed predictions
        mock_predict.return_value = [
            {"prediction": 1, "probability": 0.75},
            {"prediction": 0, "probability": 0.25},
        ]

        # Make request
        response = test_client.post("/predictions/batch", json=batch_prediction_payload)

        # Check response
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2
        assert data["predictions"][0]["prediction"] == 1
        assert data["predictions"][1]["prediction"] == 0
