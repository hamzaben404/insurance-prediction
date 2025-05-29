# tests/security/test_api_security.py
from unittest.mock import patch

import pytest


@pytest.fixture
def mock_prediction_service():
    """Mock prediction service for testing"""
    with patch("src.api.services.prediction_service.PredictionService") as MockService:
        # Configure the mock
        instance = MockService.return_value
        instance.predict.return_value = [{"prediction": 0, "probability": 0.1}]

        # Return the mock
        yield instance


def test_input_validation(test_client, mock_prediction_service):
    """Test input validation"""
    # Patch the dependency injector to use our mock
    with patch(
        "src.api.dependencies.get_prediction_service",
        return_value=mock_prediction_service,
    ):
        # Test with missing required fields
        response = test_client.post(
            "/predictions/predict",
            json={"gender": "Male"},  # Missing most required fields
        )

        # Validation should fail with 422 Unprocessable Entity
        assert response.status_code == 422


def test_invalid_json(test_client):
    """Test handling of invalid JSON"""
    response = test_client.post(
        "/predictions/predict",
        headers={"Content-Type": "application/json"},
        content="{'invalid': json",  # Invalid JSON
    )
    assert response.status_code == 422  # Unprocessable Entity
