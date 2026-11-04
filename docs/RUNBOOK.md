# Operational Runbook — Google Play Store MLOps Platform

## 1. Prerequisites and Environment Setup

Before executing pipelines or managing production deployments:
1. **Python Environment:** Ensure Python 3.11 or 3.12 is active with dependencies installed:
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Docker Engine:** Ensure Docker Desktop or the local Docker daemon is running (`docker info`).
3. **Kubernetes Cluster:** Confirm Kubernetes is accessible via `kubectl`:
   ```bash
   kubectl cluster-info
   kubectl get nodes
   ```
4. **Configuration Check:** Verify `config.yaml` exists at the project root with valid tracking URIs (`sqlite:///mlflow.db`) and quality gate thresholds.

---

## 2. Pipeline Execution Commands

### 2.1 Full MLOps Pipeline Execution
Runs all 10 stages sequentially (ingestion, schema check, preprocessing, training, tracking, evaluation, quality gate, lifecycle promotion, deployment, and inference):
```bash
python src/mlops_pipeline/play_store_pipeline.py
```

### 2.2 Continuous Drift Monitoring & Retraining Trigger
Evaluates data, prediction, and concept drift reports, inspects incoming batch availability, and conditionally launches retraining:
```bash
python src/mlops_pipeline/12_monitor_trigger.py
```

### 2.3 Local API Serving
Run the FastAPI inference service locally:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation is available at: `http://localhost:8000/docs`

### 2.4 Container Build and Kubernetes Deployment
Manually trigger Docker packaging and Kubernetes rollout:
```bash
python deployment/deploy_production.py
```

---

## 3. Interpreting Operational Reports

| Report File | Key Metric / Field | Operational Meaning |
|---|---|---|
| `artifacts/data_load_report.json` | `status: "already_ingested"` | Batch SHA-256 hash already exists in lineage. Ingestion and retraining bypassed safely. |
| `artifacts/data_load_report.json` | `status: "success"` | Novel batch successfully appended; prior dataset snapshot archived in `data/raw/archive/`. |
| `outputs/drift/drift_analysis_report.csv` | `drift_status: "DRIFT"` | Statistically significant drift detected across data, prediction, or concept dimensions. |
| `outputs/monitoring/monitor_trigger_report.json` | `trigger_decision` | `retraining_completed`, `no_drift`, `drift_waiting_for_data`, or `no_new_data_retraining_skipped`. |
| `reports/quality_gate_report.json` | `decision: "PASSED"` | Candidate achieved recall $\ge 0.70$. Eligible for lifecycle promotion comparison. |
| `reports/quality_gate_report.json` | `decision: "FAILED"` | Candidate failed SLA. Pipeline immediately halted; candidate model quarantined. |
| `reports/lifecycle_report.json` | `promoted: true` | Candidate outperformed current Production model. Promoted to Production; trigger rollout. |
| `reports/lifecycle_report.json` | `promoted: false` | Candidate did not improve on Production. Candidate archived; active Production model retained. |
| `reports/deployment_automation_report.json` | `deployment_status: "success"` | Docker image built and Kubernetes rollout completed successfully. |

---

## 4. Troubleshooting Guide

### 4.1 Ingestion Reports `already_ingested`
- **Symptom:** Orchestrator stops after Stage 1 with `new_data_ingested: false`.
- **Root Cause:** The SHA-256 hash of `data/raw/new_googleplaystore.csv` is already recorded in `data/raw/archive/dataset_lineage.json`.
- **Resolution:** This is standard idempotent behavior. To process a new batch, supply a genuinely new dataset with unique records.

### 4.2 Candidate Model Fails Quality Gate
- **Symptom:** Pipeline fails at Stage 7 with `quality_gate_report.json` showing `PASSED = false`.
- **Root Cause:** Candidate model recall is below 0.70.
- **Resolution:** Inspect feature importance in MLflow, review recent data drift reports, and tune hyperparameters (e.g. class weighting, tree depth) in `config.yaml`.

### 4.3 Deployment Step Skipped
- **Symptom:** `deploy_if_promoted.py` outputs "Candidate was not promoted. Skipping deployment."
- **Root Cause:** Expected behavior when candidate model recall does not exceed the active Production model.
- **Resolution:** No action needed; existing Production model continues serving live inference traffic.

### 4.4 Kubernetes Pods in `CrashLoopBackOff`
- **Symptom:** `kubectl get pods -l app=play-store-api` shows pods crashing.
- **Diagnostics:**
  ```bash
  kubectl logs -l app=play-store-api --tail=100
  kubectl describe pod -l app=play-store-api
  ```
- **Common Fix:** Ensure `deployment/model.pkl` is not missing or corrupt, and that the container image contains all runtime dependencies from `requirements-api.txt`.

---

## 5. Rollback and Disaster Recovery

### Reverting to a Previous Production Model Version:
If an operational regression is discovered post-rollout:
1. Open MLflow UI or run:
   ```bash
   python inspect_model_version.py
   ```
2. Transition previous stable model version back to `Production` stage in MLflow.
3. Export the model bundle to `deployment/`:
   ```bash
   python src/mlops_pipeline/11_prepare_deployment.py
   ```
4. Trigger container rebuild and rolling update:
   ```bash
   python deployment/deploy_production.py
   ```
5. Confirm healthy Kubernetes rollout:
   ```bash
   kubectl rollout status deployment/play-store-api
   ```
