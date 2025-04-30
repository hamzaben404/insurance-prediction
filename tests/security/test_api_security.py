# tests/security/test_api_security.py


def test_input_validation(test_client):
    """Test input validation"""
    # Test with missing required fields
    response = test_client.post(
        "/predictions/predict", json={"gender": "Male"}  # Missing most required fields
    )
    assert response.status_code == 422  # Unprocessable Entity

    # Test with invalid data type
    response = test_client.post(
        "/predictions/predict",
        json={
            "gender": "Male",
            "age": "not_a_number",  # Should be a number
            "has_driving_license": True,
            "region_id": 28,
            "vehicle_age": "1-2 Year",
            "past_accident": "No",
            "annual_premium": 2630.0,
            "sales_channel_id": 26,
            "days_since_created": 80,
        },
    )
    assert response.status_code == 422  # Unprocessable Entity


def test_invalid_json(test_client):
    """Test handling of invalid JSON"""
    response = test_client.post(
        "/predictions/predict",
        headers={"Content-Type": "application/json"},
        content="{'invalid': json",  # Invalid JSON
    )
    assert response.status_code == 422  # Unprocessable Entity
