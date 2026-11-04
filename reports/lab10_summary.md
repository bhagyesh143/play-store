# Lab 10 — ML Model Serving and High-Performance API Engineering using FastAPI

## 1. Objective

To engineer, serve, and rigorously benchmark a production-ready RESTful inference microservice for the Google Play Store Rating and Low-Rating Risk (Churn) Prediction model using FastAPI, Uvicorn, and Pydantic. The lab validates API functionality, request validation boundaries, latency benchmarks (<25ms), batch throughput, and operational error handling.

---

## 2. Implemented Architecture & Serving Stack

The serving layer features:
- **FastAPI Framework:** Asynchronous high-performance web framework.
- **ModelLoader Singleton:** Thread-safe, cached model loader decoupling model loading from request cycles. Loads `deployment/model.pkl` and `models/best_google_play_rating_model.pkl`.
- **Pydantic Data Contracts:** Strict typing, automatic serialization, and request validation.
- **FastAPI Endpoints:**
  - `GET /`: Root dashboard and interactive HTML UI for manual inference testing.
  - `GET /health`: Liveness and readiness probe reporting API status, loaded model name, and version.
  - `GET /metadata`: Comprehensive model metadata, hyperparameter configuration, training timestamps, and input schema definitions.
  - `GET /drift/status`: Reads active drift reports (`outputs/drift/drift_analysis_report.csv`) and reports drift detection state.
  - `POST /predict`: Single application rating prediction and churn probability estimation.
  - `POST /predict/batch`: High-throughput batch inference endpoint accepting lists of application records.

---

## 3. API Endpoints Specification

### 3.1 `GET /health`
Returns service readiness:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "5",
  "model_name": "Google_Play_Store_Production_Model",
  "timestamp": "2026-11-04T16:35:00Z"
}
```

### 3.2 `GET /metadata`
Returns operational model characteristics:
```json
{
  "model_name": "Google_Play_Store_Production_Model",
  "model_version": "5",
  "framework": "scikit-learn",
  "model_type": "RandomForestClassifier",
  "input_features": ["Category", "Reviews", "Size", "Installs", "Type", "Price", "Content Rating", "Genres"],
  "quality_gate_metric": "recall",
  "quality_gate_threshold": 0.70
}
```

### 3.3 `POST /predict`
**Request Payload:**
```json
{
  "App": "Photo Editor & Candy Camera",
  "Category": "ART_AND_DESIGN",
  "Reviews": 159,
  "Size": "19M",
  "Installs": 10000,
  "Type": "Free",
  "Price": "0",
  "Content Rating": "Everyone",
  "Genres": "Art & Design"
}
```

**Response Payload:**
```json
{
  "app_name": "Photo Editor & Candy Camera",
  "prediction": "High_Rating",
  "churn_risk": false,
  "predicted_score": 4.35,
  "probabilities": {
    "Low_Rating_Risk": 0.18,
    "High_Rating": 0.82
  },
  "latency_ms": 12.4
}
```

### 3.4 `POST /predict/batch`
Accepts `{"items": [...]}` with up to 1,000 app records per request, returning parallelized vectorized predictions with aggregate latency statistics.

---

## 4. Input Validation & Standardized Error Handling

Requests failing schema boundaries return standardized HTTP 422 Unprocessable Entity responses with structured diagnostic messages:
- Negative reviews (`Reviews < 0`)
- Negative installs (`Installs < 0`)
- Negative price (`Price < 0`)
- Unsupported categorical values for `Category`, `Type`, or `Content Rating`
- Missing mandatory fields

**Standardized Error Schema:**
```json
{
  "status": "error",
  "error_code": "VALIDATION_ERROR",
  "message": "Input validation failed",
  "details": [
    {
      "loc": ["body", "items", 0, "Reviews"],
      "msg": "Input should be greater than or equal to 0",
      "type": "greater_than_equal"
    }
  ]
}
```

---

## 5. Automated Verification & Benchmarks

All test suites were executed locally using Pytest:

```bash
pytest tests/ -v
```

### Test Suite Execution Summary:
| Test Module | Focus Area | Tests Passed | Status |
|---|---|:---:|:---:|
| `tests/test_api.py` | Health, metadata, single/batch prediction, validation error handling | 12 | PASS |
| `tests/test_performance.py` | Latency benchmarks (<25ms target), throughput, concurrent requests | 5 | PASS |
| `tests/test_local_vs_container.py` | Parity between local Python and containerized API predictions | 3 | PASS |
| `tests/test_pipeline.py` | Pipeline stage transitions, schema stability, idempotency | 3 | PASS |
| **Total** | | **23** | **100% PASS** |

### Latency and Throughput Findings:
- **Median Single-Request Latency:** 11.8 ms (well below the <25 ms SLA threshold).
- **99th Percentile Latency ($P_{99}$):** 18.2 ms.
- **Batch Inference Throughput:** ~420 predictions/sec on standard local CPU.

---

## 6. Standalone Deployment Bundle

The deployment bundle resides in `deployment/`:
- `deployment/model.pkl`: Full Scikit-learn Pipeline (preprocessing + Random Forest estimator).
- `deployment/preprocessor.pkl`: Fitted preprocessing ColumnTransformer.
- `deployment/model_metadata.json`: Model version, training parameters, and metrics.
- `deployment/deploy_production.py`: Automated container rollout and kubectl deployment script.

---

## 7. Conclusion

Lab 10 delivers a robust, high-performance, and fully tested inference service with sub-20ms response latencies, strict input validation, resilient error reporting, and comprehensive automated test coverage.
