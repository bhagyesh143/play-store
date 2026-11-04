import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os
import sys
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def configure_mlflow(config):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def train_and_register_model():
    print("[INFO] --- Starting MLflow Training Run & Model Registration ---")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    configure_mlflow(config)
    experiment_name = config["mlflow"]["experiment_name"]
    mlflow.set_experiment(experiment_name)

    processed_dir = config["data"]["processed_dir"]

    try:
        X_train = np.load(os.path.join(processed_dir, "X_train_final.npy"))
        X_test = np.load(os.path.join(processed_dir, "X_test_final.npy"))
        y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
        y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    except FileNotFoundError:
        print("[ERROR] Processed data not found. Please run 03_preprocessing.py first.")
        return False

    model_cfg = config["model"]
    params = {
        "n_estimators": model_cfg["n_estimators"],
        "max_depth": model_cfg["max_depth"],
        "random_state": model_cfg["random_state"],
        "class_weight": model_cfg["class_weight"],
    }

    with mlflow.start_run(run_name="RandomForest_PlayStore") as run:
        mlflow.log_params(params)
        mlflow.log_param("model_family", "RandomForestClassifier")
        mlflow.log_param("target", config["data"]["target_column"])
        mlflow.log_param("lab", "Lab 7-13 MLOps Pipeline")

        if os.getenv("GITHUB_RUN_ID"):
            mlflow.log_param("github_run_id", os.getenv("GITHUB_RUN_ID"))
        if os.getenv("GITHUB_SHA"):
            mlflow.log_param("github_commit", os.getenv("GITHUB_SHA"))

        print("[INFO] Training RandomForestClassifier model...")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        print("[INFO] Evaluating model on held-out test set...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        }

        mlflow.log_metrics(metrics)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, config["paths"]["model"])
        joblib.dump(model, "models/best_google_play_rating_model.pkl")

        preprocessor_path = config["paths"]["preprocessor"]
        if os.path.exists(preprocessor_path):
            mlflow.log_artifact(preprocessor_path, artifact_path="preprocessing_pipeline")

        registered_name = config["registry"]["model_name"]
        print(f"[INFO] Registering candidate model as '{registered_name}'...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest_model",
            registered_model_name=registered_name,
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )

        print(
            f"[SUCCESS] Metrics: Accuracy={metrics['accuracy']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"F1={metrics['f1_score']:.4f} | "
            f"ROC-AUC={metrics['roc_auc']:.4f}"
        )
        print(f"[SUCCESS] Model artifact saved to: {config['paths']['model']}")
        print(f"[SUCCESS] MLflow Run ID: {run.info.run_id}")

    return True


if __name__ == "__main__":
    passed = train_and_register_model()
    sys.exit(0 if passed else 1)
