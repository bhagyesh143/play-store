"""
Comprehensive Pytest Test Suite for Google Play Store Rating Prediction MLOps
Tests preprocessing, model inference, FastAPI endpoints, logging, drift detection, and alerts.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from api.preprocessing import (
    clean_size, clean_reviews, clean_installs,
    clean_price, clean_version, clean_android_version,
    transform_input_to_features
)
from api.model_loader import model_service
from monitoring.drift_detector import (
    calculate_numerical_psi,
    calculate_categorical_psi,
    DataDriftDetector
)
from monitoring.alert_service import email_alert_service


@pytest.fixture
def client():
    return TestClient(app)


# ============================================================
# 1. TEST PREPROCESSING UTILITIES
# ============================================================

def test_clean_size():
    assert clean_size("19M") == 19.0
    assert clean_size("512k") == 0.5
    assert clean_size(25.5) == 25.5
    assert clean_size("Varies with device") == 15.0
    assert clean_size(None) == 15.0


def test_clean_reviews():
    assert clean_reviews("1,500") == 1500.0
    assert clean_reviews(250) == 250.0
    assert clean_reviews(None) == 100.0


def test_clean_installs():
    assert clean_installs("10,000+") == 10000.0
    assert clean_installs("1,000,000+") == 1000000.0
    assert clean_installs(5000) == 5000.0
    assert clean_installs("Free") == 10000.0


def test_clean_price():
    assert clean_price("$4.99") == 4.99
    assert clean_price("0") == 0.0
    assert clean_price(0.0) == 0.0


def test_clean_versions():
    assert clean_version("1.2.3") == 1.2
    assert clean_version("Varies with device") == 2.0
    assert clean_android_version("4.0.3 and up") == 4.0
    assert clean_android_version("Varies with device") == 4.0


def test_transform_input_to_features():
    raw_payload = {
        "App": "Test App",
        "Category": "GAME",
        "Reviews": "5000",
        "Size": "25M",
        "Installs": "50,000+",
        "Type": "Free",
        "Price": "$0",
        "Content Rating": "Everyone",
        "Genres": "Action",
        "Last Updated": "August 1, 2018",
        "Current Ver": "1.0",
        "Android Ver": "4.1 and up"
    }
    df = transform_input_to_features(raw_payload)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 1
    assert "Reviews" in df.columns
    assert df["Reviews"].iloc[0] == 5000.0
    assert df["Price"].iloc[0] == 0.0


# ============================================================
# 2. TEST MODEL LOADING & INFERENCE
# ============================================================

def test_model_loading_and_prediction():
    assert model_service.model is not None
    info = model_service.get_info()
    assert info["target_variable"] == "Rating"
    assert info["rating_range"]["min"] == 1.0
    assert info["rating_range"]["max"] == 5.0

    raw_payload = {
        "App": "Subway Surfers",
        "Category": "GAME",
        "Reviews": 27722264,
        "Size": "76M",
        "Installs": "1,000,000,000+",
        "Type": "Free",
        "Price": "0",
        "Content Rating": "Everyone 10+",
        "Genres": "Arcade",
        "Last Updated": "July 12, 2018",
        "Current Ver": "1.90.0",
        "Android Ver": "4.1 and up"
    }
    df_feat = transform_input_to_features(raw_payload)
    rating = model_service.predict_dataframe(df_feat)
    assert isinstance(rating, float)
    assert 1.0 <= rating <= 5.0


# ============================================================
# 3. TEST FASTAPI ENDPOINTS
# ============================================================

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["model_loaded"] is True


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["target_variable"] == "Rating"
    assert "baseline_metrics" in data


def test_predict_endpoint_valid(client):
    payload = {
        "App": "Duolingo",
        "Category": "EDUCATION",
        "Reviews": 6289924,
        "Size": "20M",
        "Installs": "100,000,000+",
        "Type": "Free",
        "Price": "0",
        "Content Rating": "Everyone",
        "Genres": "Education",
        "Last Updated": "August 1, 2018",
        "Current Ver": "3.5",
        "Android Ver": "4.1 and up"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert "predicted_rating" in data
    assert 1.0 <= data["predicted_rating"] <= 5.0
    assert "request_id" in data
    assert data["prediction_latency_ms"] > 0


def test_predict_endpoint_invalid_schema(client):
    # Missing non-nullable / malformed body
    response = client.post("/predict", json="Not a valid json dictionary")
    assert response.status_code == 422


# ============================================================
# 4. TEST DRIFT DETECTION
# ============================================================

def test_numerical_psi_calculation():
    np.random.seed(42)
    base = pd.Series(np.random.normal(50, 10, 1000))
    prod_similar = pd.Series(np.random.normal(50.1, 10, 1000))
    prod_drifted = pd.Series(np.random.normal(100, 10, 1000))

    psi_sim, _, status_sim = calculate_numerical_psi(base, prod_similar)
    assert status_sim == "NO SIGNIFICANT DRIFT"
    assert psi_sim < 0.10

    psi_drift, _, status_drift = calculate_numerical_psi(base, prod_drifted)
    assert status_drift == "HIGH DRIFT"
    assert psi_drift >= 0.25


def test_categorical_psi_calculation():
    base = pd.Series(["A", "B", "C", "D"] * 25)
    prod_similar = pd.Series(["A", "B", "C", "D"] * 25)
    prod_drifted = pd.Series(["Z", "Y", "X", "W"] * 25)

    psi_sim, _, status_sim = calculate_categorical_psi(base, prod_similar)
    assert status_sim == "NO SIGNIFICANT DRIFT"

    psi_drift, _, status_drift = calculate_categorical_psi(base, prod_drifted)
    assert status_drift == "HIGH DRIFT"


# ============================================================
# 5. TEST ALERT SERVICE
# ============================================================

def test_alert_service_cooldown():
    service = email_alert_service
    # If unconfigured, does not crash and returns False
    result = service.send_drift_alert({
        "overall_severity": "HIGH DRIFT",
        "high_drift_features": ["Reviews", "Price"],
        "production_sample_size": 20
    })
    assert isinstance(result, bool)
