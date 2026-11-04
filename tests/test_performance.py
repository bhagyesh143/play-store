import time
import statistics
import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

VALID_APP = {
    "App": "Subway Surfers",
    "Category": "GAME",
    "Reviews": 27722264,
    "Size": "76M",
    "Installs": "1,000,000,000+",
    "Type": "Free",
    "Price": "0",
    "Content Rating": "Everyone 10+",
    "Genres": "Arcade",
    "Android Ver": "4.1 and up"
}


def test_single_prediction_latency_benchmark(client):
    latencies = []
    iterations = 25

    for _ in range(iterations):
        start = time.perf_counter()
        resp = client.post("/predict", json=VALID_APP)
        duration_ms = (time.perf_counter() - start) * 1000.0
        assert resp.status_code == 200
        latencies.append(duration_ms)

    avg_latency = statistics.mean(latencies)
    median_latency = statistics.median(latencies)

    print(f"\n[BENCHMARK] Average Latency: {avg_latency:.2f} ms")
    print(f"[BENCHMARK] Median Latency : {median_latency:.2f} ms")
    assert avg_latency < 250.0  # Generous threshold for local testing
    assert len(latencies) == iterations


def test_batch_throughput(client):
    batch_size = 10
    payload = {"applications": [VALID_APP] * batch_size}

    start = time.perf_counter()
    resp = client.post("/predict/batch", json=payload)
    duration_ms = (time.perf_counter() - start) * 1000.0

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_applications"] == batch_size
    assert len(data["predictions"]) == batch_size
    print(f"\n[BENCHMARK] Batch of {batch_size} completed in {duration_ms:.2f} ms")
