import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

TEST_APP = {
    "App": "Instagram",
    "Category": "SOCIAL",
    "Reviews": 66577313,
    "Size": "Varies with device",
    "Installs": "1,000,000,000+",
    "Type": "Free",
    "Price": "0",
    "Content Rating": "Teen",
    "Genres": "Social",
    "Android Ver": "Varies with device"
}


def test_prediction_consistency(client):
    """
    Validates that identical payloads produce identical predictions deterministically.
    """
    res1 = client.post("/predict", json=TEST_APP)
    res2 = client.post("/predict", json=TEST_APP)

    assert res1.status_code == 200
    assert res2.status_code == 200

    pred1 = res1.json()["predicted_rating"]
    pred2 = res2.json()["predicted_rating"]

    assert pred1 == pred2, f"Predictions should be deterministic: {pred1} != {pred2}"
