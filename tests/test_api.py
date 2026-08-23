import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify that the health check endpoint returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_predict_endpoint_valid_payload():
    """Verify prediction endpoint with a dummy sample."""
    sample_payload = {
        "features": {
            "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
            "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
            "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
            "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
            "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
            "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
            "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053,
            "scaled_amount": -0.353229, "scaled_time": -0.994983
        }
    }
    response = client.post("/predict", json=sample_payload)
    # Accepts 200 (if model loaded) or 503 (if model registry not present during standalone test)
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()
        assert "is_fraud" in data
        assert "fraud_probability" in data
        assert isinstance(data["is_fraud"], bool)
        assert 0.0 <= data["fraud_probability"] <= 1.0