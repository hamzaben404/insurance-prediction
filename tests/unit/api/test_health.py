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
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_health_endpoint_head():
    """Test health endpoint with HEAD method"""
    response = client.head("/health")
    assert response.status_code == 200


def test_model_health_endpoint():
    """Test model health endpoint"""
    response = client.get("/health/model")
    assert response.status_code == 200
    assert "status" in response.json()


def test_metrics_endpoint():
    """Test metrics endpoint"""
    response = client.get("/health/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "cpu_percent" in data
    assert "memory_percent" in data
    assert "disk_percent" in data
