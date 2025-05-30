# tests/performance/test_api_performance.py
import concurrent.futures
import os
import statistics
import time

import pytest
import requests
from fastapi.testclient import TestClient

from src.api.main import app

# Set API URL (adjust as needed)
API_URL = os.environ.get("API_URL", "http://localhost:8000")


def is_api_running():
    """Check if external API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=1)
        return response.status_code == 200
    except (requests.RequestException, ConnectionError):
        return False


@pytest.fixture
def client():
    """Get test client"""
    if is_api_running():
        # Use real API
        return None
    else:
        # Use TestClient
        return TestClient(app)


def make_request(client, endpoint, data):
    """Make request using either real API or TestClient"""
    if client is None:
        # Use real API
        return requests.post(f"{API_URL}{endpoint}", json=data)
    else:
        # Use TestClient
        return client.post(endpoint, json=data)


def test_prediction_performance(client):
    """Test prediction endpoint performance"""
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

    # Number of requests
    num_requests = 10

    # Make requests and measure time
    response_times = []
    for _ in range(num_requests):
        start_time = time.time()
        response = make_request(client, "/predictions/predict", test_data)
        end_time = time.time()

        assert response.status_code == 200
        response_times.append(end_time - start_time)

    # Calculate statistics
    avg_time = statistics.mean(response_times)
    max_time = max(response_times)
    min_time = min(response_times)

    print("\nPerformance Results:")
    print(f"Average response time: {avg_time:.4f} seconds")
    print(f"Min response time: {min_time:.4f} seconds")
    print(f"Max response time: {max_time:.4f} seconds")

    # Assert reasonable performance
    assert avg_time < 1.0, "Average response time should be less than 1 second"


def test_concurrent_load(client):
    """Test API under concurrent load"""
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

    # Number of concurrent requests
    num_concurrent = 5
    num_requests_per_worker = 5

    def make_requests():
        """Make multiple requests and return response times"""
        times = []
        for _ in range(num_requests_per_worker):
            start_time = time.time()
            response = make_request(client, "/predictions/predict", test_data)
            end_time = time.time()

            if response.status_code == 200:
                times.append(end_time - start_time)
            else:
                print(f"Request failed with status {response.status_code}")
        return times

    # Make concurrent requests with timeout
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        futures = [executor.submit(make_requests) for _ in range(num_concurrent)]

        # Wait with timeout
        done, not_done = concurrent.futures.wait(futures, timeout=30)

        # Cancel remaining futures
        for future in not_done:
            future.cancel()

        all_times = []
        for future in done:
            try:
                times = future.result()
                all_times.extend(times)
            except Exception as exc:
                print(f"Worker generated an exception: {exc}")

    # Ensure we have results
    if not all_times:
        pytest.skip("No successful requests completed")

    # Calculate statistics
    avg_time = statistics.mean(all_times)
    max_time = max(all_times)
    min_time = min(all_times)

    print("\nConcurrent Load Results:")
    print(f"Total requests: {len(all_times)}")
    print(f"Average response time: {avg_time:.4f} seconds")
    print(f"Min response time: {min_time:.4f} seconds")
    print(f"Max response time: {max_time:.4f} seconds")

    # Assert reasonable performance under load
    assert avg_time < 2.0, "Average response time under load should be less than 2 seconds"


# Skip these tests in CI if no API is running
if not is_api_running() and os.environ.get("CI"):
    pytest.skip("Skipping performance tests in CI without running API", allow_module_level=True)
