# Lab 11 — Containerization Report: Google Play Store Prediction API

## 1. Objective

The objective of Lab 11 is to package and containerize the Google Play Store ML inference microservice using Docker, validate runtime portability across environments, verify zero prediction drift between local and containerized execution, configure robust automated container health checks, and document container images for production rollout.

---

## 2. Container Architecture & Dockerfile Design

The Docker image is built using an optimized multi-stage Python runtime:
- **Base Image:** `python:3.11-slim` (minimizes image attack surface and overall footprint to ~380MB).
- **Working Directory:** `/app`
- **Dependency Isolation:** Installs only runtime serving dependencies via `requirements-api.txt` without heavy development tools.
- **Artifact Bundle:** Copies the validated standalone deployment bundle (`deployment/model.pkl`, `deployment/preprocessor.pkl`, `deployment/model_metadata.json`) and API codebase (`api/`).
- **Container Health Check:**
  ```dockerfile
  HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
  ```
- **Service Entrypoint:** Runs Uvicorn ASGI server bound to `0.0.0.0:8000`.

---

## 3. Image Build and Tagging Specifications

The container image is tagged with semantic model versioning matching the MLflow Production registry version:

```bash
docker build -t play-store-api:model-v5 -f Dockerfile .
docker tag play-store-api:model-v5 play-store-api:latest
```

### Container Execution:
```bash
docker run -d -p 8000:8000 \
  --name play-store-api \
  --restart unless-stopped \
  play-store-api:model-v5
```

---

## 4. Container Validation & Healthcheck Verification

### 4.1 Healthcheck Status
Querying container inspection demonstrates `healthy` status:
```bash
docker inspect --format='{{json .State.Health.Status}}' play-store-api
# Output: "healthy"
```

### 4.2 Endpoint Health Probe
```bash
curl -X GET http://localhost:8000/health
```
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "5",
  "model_name": "Google_Play_Store_Production_Model"
}
```

---

## 5. Parity Testing: Local vs. Containerized Inference

To verify that serialization, compiler differences, or operating system runtimes do not introduce subtle numerical divergence, parity tests were executed (`tests/test_local_vs_container.py`):
1. Test records were evaluated using the local Scikit-learn model object in Python.
2. The identical test records were submitted via HTTP to the running Docker container endpoint (`POST /predict`).
3. Predictions, class probabilities, and predicted rating scores were compared with a strict epsilon threshold ($\epsilon < 10^{-6}$).

### Parity Outcome:
- **Prediction Label Match:** 100% (0 discrepancies across all test records).
- **Probability Delta:** $\Delta P_{max} < 10^{-7}$.
- **Result:** Complete prediction parity confirmed.

---

## 6. Conclusion

The containerized Google Play Store prediction service is fully self-contained, portable, health-monitored, and verified to produce identical results to the verified training environment.
