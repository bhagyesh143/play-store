# MLflow CI/CD and Experiment Tracking Setup

## 1. Objective

To establish production-grade experiment tracking, artifact persistence, and model registry lifecycle governance for the Google Play Store Rating Prediction platform using MLflow.

---

## 2. Configuration Specifications

All MLflow tracking parameters are managed via `config.yaml`:

```yaml
mlflow:
  tracking_uri: "sqlite:///mlflow.db"
  experiment_name: "Google_Play_Store_Rating_Prediction"
  registered_model_name: "Google_Play_Store_Production_Model"
  artifact_location: "./mlruns"
```

- **Backend Store:** SQLite database (`sqlite:///mlflow.db`) providing transactional persistence for runs, parameters, metrics, and tags.
- **Artifact Store:** Local directory (`./mlruns`) housing model binaries, preprocessors, evaluation charts (confusion matrices, ROC curves), and metadata.
- **Model Registry:** Centrally registers versions of `Google_Play_Store_Production_Model`.

---

## 3. Serialization and Skops Security Integration

Modern MLflow (v2.14+) uses `skops` by default for Scikit-learn artifact persistence to prevent arbitrary code execution vulnerabilities associated with standard pickle serialization.

To support complex tree-based estimators (like `RandomForestClassifier`), our pipeline explicitly registers trusted numpy and scikit-learn types during logging:

```python
mlflow.sklearn.log_model(
    sk_model=model_pipeline,
    artifact_path="model",
    registered_model_name=config["mlflow"]["registered_model_name"],
    skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"]
)
```

---

## 4. Experiment Tracking and Metrics Suite

Each training run systematically logs:
- **Parameters:** `n_estimators`, `max_depth`, `min_samples_split`, `random_state`, `feature_count`, `training_samples`.
- **Metrics:** `accuracy`, `precision`, `recall`, `f1_score`, `roc_auc`.
- **Artifacts:**
  - `confusion_matrix.png`
  - `roc_curve.png`
  - `preprocessor.pkl`
  - Model pipeline bundle

---

## 5. Model Registry Lifecycle Governance

The model registry enforces structured stage transitions:
- **Staging:** Newly trained candidate versions passing schema and evaluation gates are initially transitioned to Staging.
- **Production:** `08_automate_lifecycle.py` benchmarks the Staging candidate against the active Production version using Recall. Only candidates delivering superior performance are transitioned to `Production`.
- **Archived:** Superseded Production versions and rejected candidate models are moved to `Archived` to preserve full auditability.

---

## 6. Continuous Integration & Best Practices

1. **Database Exclusion:** Keep `mlflow.db` and large artifact trees out of Git commits via `.gitignore`.
2. **Environment Reproducibility:** Ensure consistent Scikit-learn versions (`scikit-learn>=1.4.0`) across training and CI environments to avoid deserialization mismatch.
3. **Automated Auditing:** Query registered model versions using utilities like `inspect_model_version.py` and `check_mlflow.py` inside automated workflows.
