# tests/performance/test_api_performance.py
import concurrent.futures
import os
import statistics
import sys
import time

import pytest
import requests

# Set API URL (adjust as needed)
API_URL = os.environ.get("API_URL", "http://localhost:8000")


def test_prediction_performance():
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
        response = requests.post(f"{API_URL}/predictions/predict", json=test_data)
        end_time = time.time()

        assert response.status_code == 200
        response_times.append(end_time - start_time)

    # Calculate statistics
    avg_time = statistics.mean(response_times)
    max_time = max(response_times)
    min_time = min(response_times)

    print(f"\nPerformance Results:")
    print(f"Average response time: {avg_time:.4f} seconds")
    print(f"Min response time: {min_time:.4f} seconds")
    print(f"Max response time: {max_time:.4f} seconds")

    # Assert reasonable performance
    assert avg_time < 1.0, "Average response time should be less than 1 second"


def test_concurrent_load():
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
            response = requests.post(f"{API_URL}/predictions/predict", json=test_data)
            end_time = time.time()

            assert response.status_code == 200
            times.append(end_time - start_time)
        return times

    # Make concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        future_to_worker = {
            executor.submit(make_requests): i for i in range(num_concurrent)
        }

        all_times = []
        for future in concurrent.futures.as_completed(future_to_worker):
            worker_id = future_to_worker[future]
            try:
                times = future.result()
                all_times.extend(times)
            except Exception as exc:
                print(f"Worker {worker_id} generated an exception: {exc}")

    # Calculate statistics
    avg_time = statistics.mean(all_times)
    max_time = max(all_times)
    min_time = min(all_times)

    print(f"\nConcurrent Load Results:")
    print(f"Total requests: {num_concurrent * num_requests_per_worker}")
    print(f"Average response time: {avg_time:.4f} seconds")
    print(f"Min response time: {min_time:.4f} seconds")
    print(f"Max response time: {max_time:.4f} seconds")

    # Assert reasonable performance under load
    assert (
        avg_time < 2.0
    ), "Average response time under load should be less than 2 seconds"
