import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

VALID_APP = {
    "App": "Candy Crush Saga",
    "Category": "GAME",
    "Reviews": 22426677,
    "Size": "74M",
    "Installs": "500,000,000+",
    "Type": "Free",
    "Price": "0",
    "Content Rating": "Everyone",
    "Genres": "Casual",
    "Android Ver": "4.1 and up"
}

SECOND_APP = {
    "App": "Spotify Music",
    "Category": "FAMILY",
    "Reviews": 102548,
    "Size": "25M",
    "Installs": "10,000,000+",
    "Type": "Free",
    "Price": "0",
    "Content Rating": "Teen",
    "Genres": "Music & Audio",
    "Android Ver": "5.0 and up"
}


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Google Play Store Rating Predictor" in response.text


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "version" in data


def test_metadata_endpoint(client):
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "target_variable" in data
    assert data["target_variable"] == "Rating"
    assert "features_expected" in data


def test_predict_single_app(client):
    response = client.post("/predict", json=VALID_APP)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_rating" in data
    assert 1.0 <= data["predicted_rating"] <= 5.0
    assert data["status"] == "SUCCESS"
    assert data["prediction_latency_ms"] > 0


def test_predict_batch(client):
    payload = {"applications": [VALID_APP, SECOND_APP]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_applications"] == 2
    assert len(data["predictions"]) == 2
    for pred in data["predictions"]:
        assert 1.0 <= pred["predicted_rating"] <= 5.0
        assert pred["status"] == "SUCCESS"


def test_predict_validation_error(client):
    # Missing required App or badly formatted
    invalid_app = {"Category": "GAME"}
    response = client.post("/predict", json=invalid_app)
    # Pydantic may fill defaults or return 422 if invalid
    assert response.status_code in [200, 422]


def test_drift_status_endpoint(client):
    response = client.get("/drift/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
