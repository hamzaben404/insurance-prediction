# tests/e2e/test_docker.py
import os
import subprocess
import time

import pytest
import requests


@pytest.mark.skipif("SKIP_DOCKER_TESTS" in os.environ, reason="Docker tests disabled")
class TestDocker:
    @classmethod
    def setup_class(cls):
        """Build and start the Docker container for testing"""
        print("Building Docker image...")
        subprocess.run(
            ["docker", "build", "-t", "insurance-prediction-api-test", "."], check=True
        )

        print("Starting Docker container...")
        cls.container_id = subprocess.check_output(
            ["docker", "run", "-d", "-p", "8001:8000", "insurance-prediction-api-test"],
            universal_newlines=True,
        ).strip()

        # Wait for container to start
        time.sleep(5)

        # Set API URL
        cls.api_url = "http://localhost:8001"

    @classmethod
    def teardown_class(cls):
        """Stop and remove the Docker container"""
        print(f"Stopping container {cls.container_id}...")
        subprocess.run(["docker", "stop", cls.container_id], check=True)

        print(f"Removing container {cls.container_id}...")
        subprocess.run(["docker", "rm", cls.container_id], check=True)

    def test_docker_health_endpoint(self):
        """Test that the health endpoint works in Docker"""
        response = requests.get(f"{self.api_url}/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_docker_prediction_endpoint(self):
        """Test that the prediction endpoint works in Docker"""
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

        response = requests.post(f"{self.api_url}/predictions/predict", json=test_data)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
