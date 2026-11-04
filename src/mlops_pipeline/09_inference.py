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
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient


def configure_mlflow(config):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def clean_dataframe_for_inference(df):
    df = df.copy()

    # Clean numeric features
    if "Reviews" in df.columns:
        df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce").fillna(0)

    if "Size" in df.columns:
        def parse_size(val):
            val_str = str(val).strip()
            if val_str.endswith("M"):
                try:
                    return float(val_str[:-1])
                except:
                    return np.nan
            elif val_str.endswith("k"):
                try:
                    return float(val_str[:-1]) / 1024.0
                except:
                    return np.nan
            else:
                return np.nan

        df["Size"] = df["Size"].apply(parse_size)
        df["Size"] = df["Size"].fillna(15.0)

    if "Installs" in df.columns:
        df["Installs"] = (
            df["Installs"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("+", "", regex=False)
        )
        df["Installs"] = pd.to_numeric(df["Installs"], errors="coerce").fillna(10000)

    if "Price" in df.columns:
        df["Price"] = (
            df["Price"]
            .astype(str)
            .str.replace("$", "", regex=False)
        )
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0.0)

    feature_cols = [c for c in df.columns if c not in ["App", "Rating", "churn", "Last Updated", "Current Ver", "Android Ver"]]
    return df[feature_cols]


def run_inference_demo():
    print("[INFO] Starting ML Pipeline Production Inference...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    configure_mlflow(config)
    model_name = config["registry"]["model_name"]

    model = None
    try:
        client = MlflowClient()
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if prod_versions:
            prod_version = prod_versions[0]
            print(f"[INFO] Found MLflow Production version {prod_version.version} (Run ID: {prod_version.run_id})")
            model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
            print("[SUCCESS] Production model loaded from MLflow Model Registry.")
    except Exception as e:
        print(f"[INFO] Registry load notice: {e}. Falling back to saved model file.")

    if model is None:
        model_path = config["paths"]["model"]
        if not os.path.exists(model_path):
            print(f"[ERROR] Local model not found: {model_path}")
            return False
        model = joblib.load(model_path)
        print(f"[SUCCESS] Loaded fallback model from {model_path}")

    preprocessor_path = config["paths"]["preprocessor"]
    raw_data_path = config["data"]["raw_path"]

    if not os.path.exists(preprocessor_path) or not os.path.exists(raw_data_path):
        print("[ERROR] Required preprocessor or dataset missing.")
        return False

    preprocessor = joblib.load(preprocessor_path)
    df = pd.read_csv(raw_data_path).dropna(subset=["Rating"])
    sample = df.sample(n=min(5, len(df)), random_state=42)

    X_sample_clean = clean_dataframe_for_inference(sample)
    X_sample_trans = preprocessor.transform(X_sample_clean)

    predictions = model.predict(X_sample_trans)
    probabilities = model.predict_proba(X_sample_trans)[:, 1]

    result = sample.copy()
    result["Predicted_Churn"] = predictions
    result["Churn_Probability"] = np.round(probabilities, 4)

    print("\n--- ML Pipeline Production Inference Results ---")
    display_cols = ["App", "Category", "Rating", "Predicted_Churn", "Churn_Probability"]
    print(result[[c for c in display_cols if c in result.columns]].to_string(index=False))
    print("------------------------------------------------\n")

    os.makedirs("reports", exist_ok=True)
    output_path = config["paths"]["inference_output"]
    result.to_csv(output_path, index=False)
    print(f"[SUCCESS] Predictions saved to: {output_path}")
    return True


if __name__ == "__main__":
    passed = run_inference_demo()
    sys.exit(0 if passed else 1)
