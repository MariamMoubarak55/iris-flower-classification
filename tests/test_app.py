"""Unit tests for Flask API endpoints (Slide 20)."""

import pytest
from src.template_app.app import create_app
from src.main.train import run_training_pipeline


@pytest.fixture(scope="module")
def client():
    """Create Flask test client and ensure model is trained."""
    # Ensure model is trained first
    run_training_pipeline()
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home_endpoint(client):
    """Test GET / returns online status."""
    response = client.get("/")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "online"


def test_health_endpoint(client):
    """Test GET /health returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "healthy"


def test_predict_endpoint_success(client):
    """Test POST /predict with valid features."""
    payload = {
        "features": [
            [5.1, 3.5, 1.4, 0.2]
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["prediction_name"] == "setosa"


def test_predict_endpoint_dict_format(client):
    """Test POST /predict with named key-value dictionary."""
    payload = {
        "sepal_length": 6.7,
        "sepal_width": 3.0,
        "petal_length": 5.2,
        "petal_width": 2.3
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["predictions"][0]["prediction_name"] == "virginica"


def test_predict_endpoint_bad_request(client):
    """Test POST /predict with bad input returns 400 error."""
    response = client.post("/predict", json={"unknown_key": 123})
    assert response.status_code == 400
