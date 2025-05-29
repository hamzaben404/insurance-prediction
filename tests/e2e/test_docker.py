# tests/e2e/test_docker.py
import os
import subprocess
import time

import pytest
import requests


def is_docker_available():
    """Check if Docker is available"""
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def is_ci_environment():
    """Check if running in CI environment"""
    return os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(
    not is_docker_available() or is_ci_environment() or "SKIP_DOCKER_TESTS" in os.environ,
    reason="Docker not available, running in CI, or Docker tests disabled",
)
class TestDocker:
    @classmethod
    def setup_class(cls):
        """Build and start the Docker container for testing"""
        print("Building Docker image...")
        try:
            subprocess.run(
                ["docker", "build", "-t", "insurance-prediction-api-test", "."],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Docker build failed: {e.stderr.decode()}")
            raise

        print("Starting Docker container...")
        try:
            cls.container_id = subprocess.check_output(
                ["docker", "run", "-d", "-p", "8001:8000", "insurance-prediction-api-test"],
                universal_newlines=True,
            ).strip()
        except subprocess.CalledProcessError as e:
            print(f"Docker run failed: {e}")
            raise

        # Wait for container to be ready
        cls.api_url = "http://localhost:8001"
        max_retries = 30
        for i in range(max_retries):
            try:
                response = requests.get(f"{cls.api_url}/health", timeout=1)
                if response.status_code == 200:
                    print("Container is ready!")
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)

            # Check if container is still running
            try:
                result = subprocess.run(
                    ["docker", "ps", "-q", "-f", f"id={cls.container_id}"],
                    capture_output=True,
                    text=True,
                )
                if not result.stdout.strip():
                    # Container stopped, get logs
                    logs = subprocess.check_output(
                        ["docker", "logs", cls.container_id], universal_newlines=True
                    )
                    print(f"Container logs:\n{logs}")
                    raise RuntimeError("Container stopped unexpectedly")
            except subprocess.CalledProcessError:
                pass
        else:
            # Get container logs for debugging
            try:
                logs = subprocess.check_output(
                    ["docker", "logs", cls.container_id], universal_newlines=True
                )
                print(f"Container logs:\n{logs}")
            except subprocess.CalledProcessError:
                pass
            raise RuntimeError("Container failed to start within timeout")

    @classmethod
    def teardown_class(cls):
        """Stop and remove the Docker container"""
        if hasattr(cls, "container_id"):
            print(f"Stopping container {cls.container_id}...")
            subprocess.run(["docker", "stop", cls.container_id], capture_output=True)

            print(f"Removing container {cls.container_id}...")
            subprocess.run(["docker", "rm", cls.container_id], capture_output=True)

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


# Test for CI environment - just verify files exist
def test_docker_files_exist():
    """Test that Docker-related files exist"""
    assert os.path.exists("Dockerfile")
    assert os.path.exists("docker-compose.yml")
    assert os.path.exists("requirements.txt")
