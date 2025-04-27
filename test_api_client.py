# test_api_client.py
import requests
import json
import argparse
import pytest
import argparse

@pytest.fixture
def base_url():
    """Provide base URL for API tests"""
    return "http://localhost:8000"

def test_health(base_url):
    """Test health endpoints"""
    print("Testing health endpoints...")
    
    # Test root endpoint
    response = requests.get(f"{base_url}/")
    print(f"Root endpoint: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()
    
    # Test health endpoint
    response = requests.get(f"{base_url}/health")
    print(f"Health endpoint: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()
    
    # Test model health endpoint
    response = requests.get(f"{base_url}/health/model")
    print(f"Model health endpoint: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()
    
    # Test metrics endpoint
    response = requests.get(f"{base_url}/health/metrics")
    print(f"Metrics endpoint: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_prediction(base_url):
    """Test prediction endpoints"""
    print("Testing prediction endpoints...")
    
    # Create sample input
    sample_input = {
        "gender": "Male",
        "age": 35.0,
        "has_driving_license": True,
        "region_id": 28,
        "vehicle_age": "1-2 Year",
        "past_accident": "No",
        "annual_premium": 2630.0,
        "sales_channel_id": 26,
        "days_since_created": 80
    }
    
    # Test single prediction
    response = requests.post(
        f"{base_url}/predictions/predict",
        json=sample_input
    )
    print(f"Single prediction endpoint: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()
    
    # Test batch prediction
    batch_input = {
        "records": [
            sample_input,
            {
                "gender": "Female",
                "age": 45.0,
                "has_driving_license": True,
                "region_id": 28,
                "vehicle_age": "< 1 Year",
                "past_accident": "Yes",
                "annual_premium": 3200.0,
                "sales_channel_id": 26,
                "days_since_created": 120
            }
        ]
    }
    
    response = requests.post(
        f"{base_url}/predictions/batch",
        json=batch_input
    )
    print(f"Batch prediction endpoint: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def main():
    parser = argparse.ArgumentParser(description="Test the Insurance Prediction API")
    parser.add_argument(
        "--url", type=str, default="http://localhost:8000",
        help="Base URL of the API"
    )
    args = parser.parse_args()
    
    base_url = args.url
    
    try:
        test_health(base_url)
        test_prediction(base_url)
        print("All tests completed successfully!")
    except Exception as e:
        print(f"Error testing API: {e}")

if __name__ == "__main__":
    main()