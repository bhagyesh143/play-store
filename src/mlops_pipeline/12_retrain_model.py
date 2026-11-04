import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import os
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.mlops_pipeline.inference_helper import build_preprocessor

UPDATED_DATASET_PATH = Path("data/processed/updated_training_data.csv")
TEST_FEATURES_PATH = Path("data/processed/X_test_raw.csv")
TEST_LABELS_PATH = Path("data/processed/y_test.npy")

MODEL_V2_PATH = Path("models/retrained_model_v2.pkl")
PREPROCESSOR_V2_PATH = Path("models/retrained_preprocessor_v2.pkl")
REPORT_PATH = Path("outputs/retraining/retraining_report.json")


def retrain_model():
    print("[INFO] Starting Retraining Pipeline on Updated Dataset...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    df = pd.read_csv(UPDATED_DATASET_PATH)
    target_col = config["data"].get("target_column", "churn")

    X_train = df.drop(columns=[target_col])
    y_train = df[target_col]

    X_test = pd.read_csv(TEST_FEATURES_PATH)
    y_test = np.load(TEST_LABELS_PATH)

    preprocessor, cat_cols, num_cols = build_preprocessor(X_train)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    model_cfg = config["model"]
    params = {
        "n_estimators": model_cfg["n_estimators"] + 50,
        "max_depth": model_cfg["max_depth"] + 2,
        "random_state": 42,
        "class_weight": "balanced"
    }

    with mlflow.start_run(run_name="RandomForest_Retrained_v2") as run:
        mlflow.log_params(params)
        mlflow.log_param("retrained", True)

        model = RandomForestClassifier(**params)
        model.fit(X_train_trans, y_train)

        preds = model.predict(X_test_trans)
        probs = model.predict_proba(X_test_trans)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1_score": float(f1_score(y_test, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probs)),
        }
        mlflow.log_metrics(metrics)

        MODEL_V2_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_V2_PATH)
        joblib.dump(preprocessor, PREPROCESSOR_V2_PATH)

        report = {
            "stage": "Lab 12 - Automated Retraining",
            "run_id": run.info.run_id,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "metrics": {k: round(v, 4) for k, v in metrics.items()},
            "model_path": str(MODEL_V2_PATH),
            "preprocessor_path": str(PREPROCESSOR_V2_PATH)
        }

        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        print("[SUCCESS] Retrained candidate model v2 saved.")
        print(f"[SUCCESS] Metrics: {metrics}")
        return True


if __name__ == "__main__":
    retrain_model()
