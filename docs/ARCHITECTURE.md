# MLOps Google Play Store App Rating Prediction — Architecture

## Purpose

This repository implements a production-grade Google Play Store application rating prediction and low-rating/churn risk MLOps platform with automated data validation, reproducible preprocessing, MLflow experiment tracking, model registry, quality gates, automated lifecycle promotion, multi-type drift detection, scheduled monitoring triggers, containerized serving, and Kubernetes orchestration.

## End-to-End System Architecture

```text
GitHub Actions / Scheduled Automation
              |
              v
Self-Hosted / Cloud Automation Host
              |
              v
12_monitor_trigger.py (Continuous Drift Engine)
   |          |
   |          +--> Read existing drift reports:
   |               - Data Drift (Numerical KS & Categorical JS divergence)
   |               - Prediction Drift (Output & probability KS statistics)
   |               - Concept Drift (Performance & metric degradation)
   |
   +--> Check for incoming batch (data/raw/new_googleplaystore.csv)
              |
              v
play_store_pipeline.py (Invoked when drift and new batch exist)
              |
              v
01 Data Loading + SHA-256 Batch Hashing & Lineage Deduplication
              |
              v
02 Raw Data Validation (Pandera Schema & Boundary Checks)
              |
              v
03 Reproducible Preprocessing (ColumnTransformer + Stratified Split)
              |
              v
04 Processed Output Validation (Array Shapes, Non-Empty, Zero NaNs)
              |
              v
05 Training + MLflow Experiment Tracking + Model Registration
              |
              v
06 Model Evaluation (Confusion Matrix, ROC Curve, Multi-Metric Suite)
              |
              v
07 Quality Gate Enforcement (Configured Recall Threshold >= 0.70)
              |
              v
08 Automated Model Lifecycle Manager (Production vs Candidate Comparison)
              |
        +-----+----------------+
        |                      |
   Not Promoted             Promoted
        |                      |
        v                      v
Keep Production Model    Prepare Production Deployment Bundle
(Deployment Skipped)           |
                               v
                       Build/Tag Docker Image + Kubernetes Rollout
              |
              v
09 Production Inference Demo & JSONL Audit Logging
```

## Component Matrix

| Stage | Script | Purpose | Output Artifacts |
|---|---|---|---|
| **01 Ingestion** | `01_load_data.py` | Load baseline, check incoming batch hash against lineage, archive previous snapshot, append rows | `artifacts/data_load_report.json`, `data/raw/archive/dataset_lineage.json` |
| **02 Validation** | `02_validate_data.py` | Enforce Pandera schema, column presence, type checks, and rating boundary checks | `artifacts/baseline_validation.csv` |
| **03 Preprocessing** | `03_preprocessing.py` | Imputation, scaling, one-hot encoding, stratified train/test split, persistent arrays | `models/preprocessor.pkl`, `data/processed/*.npy`, `dataset_metadata.json` |
| **04 Verification** | `04_validate_outputs.py` | Validate processed array integrity, feature dimensions, zero NaNs | `artifacts/output_validation_report.csv` |
| **05 Training & MLflow** | `05_train_registry.py` | Train RandomForest, log params/metrics to SQLite MLflow tracking, register candidate | `models/random_forest_model.pkl`, MLflow Model Registry candidate |
| **06 Evaluation** | `06_model_evaluation.py` | Comprehensive held-out test split evaluation, confusion matrix and ROC curves | `reports/model_evaluation.json`, `artifacts/confusion_matrix.png`, `roc_curve.png` |
| **07 Quality Gate** | `07_quality_gate.py` | Gate candidate models against configured threshold (`recall >= 0.70`) | `reports/quality_gate_report.json` |
| **08 Lifecycle** | `08_automate_lifecycle.py` | Compare candidate against current Production model; promote champion or archive | `reports/lifecycle_report.json` |
| **Rollout Gate** | `deploy_if_promoted.py` | Conditional deployment execution only when promotion occurs | Deployment rollout / skip status |
| **09 Inference** | `09_inference.py` | Production model inference on real app samples with probability estimates | `reports/inference_results.csv` |
| **10 Resilience** | `10_failure_injection.py` | Simulate 5 critical pipeline failure scenarios and verify recovery | `artifacts/failure_simulation_report.json`, `corrupted_simulation_errors.csv` |
| **11 Deployment** | `11_prepare_deployment.py` | Package Production model, preprocessor, and metadata into standalone bundle | `deployment/model.pkl`, `preprocessor.pkl`, `model_metadata.json` |
| **12 Monitoring** | `12_data_drift.py` | Kolmogorov-Smirnov test on numerical features, Jensen-Shannon on categoricals | `outputs/drift/data_drift_report.csv` |
| **12 Prediction Drift** | `12_prediction_drift.py` | Statistical hypothesis testing on predicted rating and probability distributions | `outputs/drift/prediction_drift_report.csv` |
| **12 Concept Drift** | `12_concept_drift.py` | Track performance degradation (accuracy, precision, recall, F1, ROC-AUC) | `outputs/drift/concept_drift_report.csv` |
| **12 Synthesis** | `12_drift_analysis.py` | Aggregates all drift dimensions into consolidated decision matrix | `outputs/drift/drift_analysis_report.csv` |
| **12 Retraining** | `12_retrain_model.py` | Trains candidate model v2 on updated drifted data batches | `models/retrained_model_v2.pkl`, `outputs/retraining/retraining_report.json` |
| **12 Benchmarking** | `12_model_comparison.py` | Direct performance benchmark comparing Production v1 and Candidate v2 | `outputs/retraining/model_comparison_report.csv`, `.json` |
| **12 Registry Update**| `12_update_mlflow_registry.py`| Updates MLflow registry stages based on retraining benchmark | `outputs/retraining/mlflow_registry_update_report.json` |
| **13 Trigger** | `12_monitor_trigger.py` | Automated trigger evaluating drift status and new batch availability | `outputs/monitoring/monitor_trigger_report.json` |
