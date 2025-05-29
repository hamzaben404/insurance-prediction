# tests/e2e/test_api_e2e.py
import atexit
import os
import subprocess
import sys
import time

import pytest
import requests

# Variable to store server process
server_process = None


@pytest.fixture(scope="module")
def api_url():
    """Start API server for testing and tear down after tests"""
    global server_process

    # Only start server if not running in Docker
    if os.environ.get("RUNNING_IN_DOCKER") != "true":
        # Start server
        print("Starting test API server...")

        # Set up environment for test
        env = os.environ.copy()
        env["MODEL_PATH"] = os.path.abspath("models/comparison/production/production_model.pkl")

        # Start server in background
        server_process = subprocess.Popen(
            [sys.executable, "-m", "src.api.server"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Register cleanup function
        def cleanup():
            if server_process:
                print("Stopping test API server...")
                server_process.terminate()
                try:
                    server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server_process.kill()

        atexit.register(cleanup)

        # Wait for server to start
        time.sleep(3)

    # Return URL
    return "http://localhost:8000"


def test_e2e_health(api_url):
    """End-to-end test for health endpoints"""
    # Test root endpoint
    response = requests.get(f"{api_url}/")
    assert response.status_code == 200

    # Test health endpoint
    response = requests.get(f"{api_url}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_e2e_prediction(api_url):
    """End-to-end test for prediction endpoint"""
    # Test data
    test_data = {
        "gender": "Male",
        "age": 35.0,
        "has_driving_license": True,
        "region_id": 28,
        "vehicle_age": "1-2 Year",
        "past_accident": "No",
        "annual_premium": 2630.0,
        "sales_channel_id": 26,
        "days_since_created": 80,
    }

    # Make request
    response = requests.post(f"{api_url}/predictions/predict", json=test_data)

    # Assert response
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert isinstance(data["prediction"], int)
    assert isinstance(data["probability"], float)
