# tests/unit/api/test_health.py


def test_root_endpoint(test_client):
    """Test root endpoint"""
    response = test_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_endpoint(test_client):
    """Test health endpoint"""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_metrics_endpoint(test_client):
    """Test metrics endpoint"""
    response = test_client.get("/health/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "cpu_percent" in data
    assert "memory_percent" in data
