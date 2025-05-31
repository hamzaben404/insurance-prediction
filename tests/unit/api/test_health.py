# tests/unit/api/test_health.py

"""Test health endpoints"""
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Insurance Prediction API"


def test_health_endpoint():
    """Test health endpoint"""
    response = client.get("/health")  # This assumes your health.router is at /health
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_health_endpoint_head():
    """Test health endpoint with HEAD method"""
    response = client.head("/health")  # This assumes your health.router is at /health
    assert response.status_code == 200


def test_model_health_endpoint():
    """Test model health endpoint"""
    # This test depends on how your model health is structured within health.router
    # Assuming it's /health/model as per your original test
    response = client.get("/health/model")
    assert response.status_code == 200
    assert "status" in response.json()  # Or specific keys related to model health


def test_metrics_endpoint():
    """Test metrics endpoint which serves Prometheus-style metrics"""
    response = client.get("/monitor/metrics")  # UPDATED PATH
    assert response.status_code == 200
    # Check for content-type, Prometheus usually uses text/plain
    assert "text/plain" in response.headers["content-type"]  # UPDATED CHECK

    # Check for presence of expected metric names in the plain text response
    # These metric names come from your src/api/routers/monitoring.py
    metrics_text = response.text
    assert "api_cpu_usage_percent" in metrics_text  # UPDATED CHECK
    assert "api_memory_usage_percent" in metrics_text  # UPDATED CHECK
    assert "api_memory_available_bytes" in metrics_text  # UPDATED CHECK
    # You can add more assertions for the structure if needed, e.g., lines starting with # HELP or # TYPE
    assert "# HELP api_cpu_usage_percent" in metrics_text
    assert "# TYPE api_cpu_usage_percent gauge" in metrics_text
