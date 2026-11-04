# LAB 7: Google Play Store MLOps Pipeline

## 1. Objective

To develop a modular, reproducible, and deterministic machine learning pipeline for Google Play Store app rating estimation and low-rating risk (churn proxy) classification. The pipeline automates the entire ML lifecycle from raw data ingestion and schema validation to reproducible preprocessing, model training, MLflow experiment tracking, automated evaluation, quality gate enforcement, lifecycle management, and production inference.

---

## 2. Pipeline Architecture & Stages

The pipeline is orchestrated by `src/mlops_pipeline/play_store_pipeline.py` (and `src/mlops_pipeline/13_end_to_end_pipeline.py`), which executes 10 sequential, modular stages with fail-fast error handling:

| Stage | Script | Core Responsibility | Primary Output |
|:---:|---|---|---|
| **1** | `01_load_data.py` | Ingests data, calculates SHA-256 batch hashes, verifies lineage against `dataset_lineage.json` for idempotent deduplication, archives snapshots, and appends valid incoming batches. | `artifacts/data_load_report.json`, `data/raw/archive/dataset_lineage.json` |
| **2** | `02_validate_data.py` | Enforces Pandera schema checks (Rating ∈ [1.0, 5.0], Reviews >= 0, Installs >= 0, Price >= 0.0, valid Categories). | `artifacts/baseline_validation.csv` |
| **3** | `03_preprocessing.py` | Parses raw strings (Size in M/k, Installs '+', Price '$'), median numerical imputation, categorical one-hot encoding, stratified train/test split. | `models/preprocessor.pkl`, `X_train_final.npy`, `X_test_final.npy`, `y_train.npy`, `y_test.npy` |
| **4** | `04_validate_outputs.py` | Verifies processed feature array dimensions, absence of NaN/infinite values, and consistent target alignment. | `artifacts/output_validation_report.csv` |
| **5** | `05_train_registry.py` | Trains Random Forest classifier/pipeline, tracks hyperparameters and metrics in SQLite MLflow backend (`sqlite:///mlflow.db`), registers candidate model in MLflow Registry. | `models/best_google_play_rating_model.pkl`, MLflow Run & Registered Model Version |
| **6** | `06_model_evaluation.py` | Computes comprehensive classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC) on the held-out test split. | `reports/model_evaluation.json`, `artifacts/confusion_matrix.png`, `artifacts/roc_curve.png` |
| **7** | `07_quality_gate.py` | Validates model performance against strict SLA requirements (Recall >= 0.70). Halts pipeline if candidate fails SLA. | `reports/quality_gate_report.json` |
| **8** | `08_automate_lifecycle.py` | Compares candidate recall against current MLflow Production model. Promotes superior candidate to Production; otherwise archives candidate. | `reports/lifecycle_report.json` |
| **9** | `deploy_if_promoted.py` | Promotion-gated deployment trigger. If promoted, extracts model, preprocessor, and metadata into standalone `deployment/` bundle. | `deployment/model.pkl`, `deployment/preprocessor.pkl`, `deployment/model_metadata.json` |
| **10** | `09_inference.py` | Executes batch prediction test against active Production model, recording output labels and probabilities. | `reports/inference_results.csv` |

---

## 3. Tools and Technologies

- **Language & Runtime:** Python 3.11 / 3.12
- **Data Engineering:** Pandas, NumPy, Scikit-learn
- **Schema Validation:** Pandera
- **Experiment Tracking & Model Registry:** MLflow (SQLite backend: `sqlite:///mlflow.db`, Skops trusted types)
- **Serialization:** Joblib, Skops, Pickle
- **Visualization:** Matplotlib, Seaborn
- **Version Control & CI/CD:** Git, GitHub Actions

---

## 4. Execution Flow & Idempotency

### Execution Command:
```bash
python src/mlops_pipeline/play_store_pipeline.py
```

### Idempotency & Batch Deduplication
If an incoming batch `data/raw/new_googleplaystore.csv` has already been ingested, `01_load_data.py` detects the matching SHA-256 hash in `data/raw/archive/dataset_lineage.json`. The orchestrator records `status: passed` and exits gracefully with `new_data_ingested: false`, preventing unnecessary re-training and compute overhead.

---

## 5. Output Artifacts and Logging

All pipeline runs record timestamped stage statuses in `logs/play_store_pipeline_run_log.json`:
- `artifacts/data_load_report.json`
- `artifacts/baseline_validation.csv`
- `artifacts/output_validation_report.csv`
- `artifacts/confusion_matrix.png`
- `artifacts/roc_curve.png`
- `reports/model_evaluation.json`
- `reports/quality_gate_report.json`
- `reports/lifecycle_report.json`
- `reports/inference_results.csv`
