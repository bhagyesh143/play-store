import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import os
import shutil
import tempfile
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient


def load_config():
    project_root = Path(__file__).resolve().parents[2]
    with open(project_root / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return project_root, config


def prepare_deployment():
    project_root, config = load_config()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"]["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    model_name = config["registry"]["model_name"]

    print("[INFO] Preparing Production Deployment Bundle...")
    print(f"[INFO] MLflow tracking URI: {tracking_uri}")
    print(f"[INFO] Target model: {model_name}")

    production_version = None
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        prod_versions = [v for v in versions if v.current_stage == "Production"]
        if prod_versions:
            production_version = prod_versions[0]
        elif versions:
            # Fallback to latest version
            production_version = max(versions, key=lambda v: int(v.version))
    except Exception as e:
        print(f"[INFO] Registry query notice: {e}")

    version_str = str(production_version.version) if production_version else "1"
    run_id = production_version.run_id if production_version else "local_run"
    stage_str = production_version.current_stage if production_version else "Production"

    model_src = project_root / config["paths"]["model"]
    preprocessor_src = project_root / config["paths"]["preprocessor"]

    if not model_src.exists():
        model_src = project_root / "models/best_google_play_rating_model.pkl"

    if not model_src.exists() or not preprocessor_src.exists():
        raise FileNotFoundError(f"Model ({model_src}) or Preprocessor ({preprocessor_src}) not found.")

    model = joblib.load(model_src)
    preprocessor = joblib.load(preprocessor_src)

    # Validate compatibility on sample
    raw_path = project_root / config["data"]["raw_path"]
    df = pd.read_csv(raw_path).dropna(subset=["Rating"])
    sample = df.head(5)

    from src.mlops_pipeline.inference_helper import clean_dataframe_for_inference
    X_clean = clean_dataframe_for_inference(sample)
    transformed = preprocessor.transform(X_clean)

    predictions = model.predict(transformed)
    if len(predictions) != len(sample):
        raise RuntimeError("Model verification failed on sample batch.")

    deployment_dir = project_root / "deployment"
    temp_dir = Path(tempfile.mkdtemp(prefix="deployment_staging_", dir=str(project_root)))
    backup_dir = None

    try:
        model_dst = temp_dir / "model.pkl"
        preprocessor_dst = temp_dir / "preprocessor.pkl"
        pipeline_candidate = project_root / "models/best_google_play_rating_model.pkl"
        if pipeline_candidate.exists():
            shutil.copy(pipeline_candidate, model_dst)
        else:
            joblib.dump(model, model_dst)
        joblib.dump(preprocessor, preprocessor_dst)

        metadata = {
            "model_name": model_name,
            "model_version": version_str,
            "model_stage": stage_str,
            "run_id": run_id,
            "source_model_path": str(model_src.relative_to(project_root)),
            "source_preprocessor_path": str(preprocessor_src.relative_to(project_root)),
            "tracking_uri": tracking_uri,
            "expected_features": int(transformed.shape[1]),
        }

        deploy_script = project_root / "deployment" / "deploy_production.py"
        if deploy_script.exists():
            shutil.copy(deploy_script, temp_dir / "deploy_production.py")

        with open(temp_dir / "model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        if deployment_dir.exists():
            idx = 1
            while True:
                candidate = project_root / f"deployment_backup_{idx}"
                if not candidate.exists():
                    backup_dir = candidate
                    break
                idx += 1
            deployment_dir.rename(backup_dir)
            print(f"[INFO] Previous deployment archived to: {backup_dir}")

        temp_dir.rename(deployment_dir)

        print("\n========================================")
        print("[SUCCESS] DEPLOYMENT PREPARATION COMPLETED")
        print("========================================")
        print("deployment/")
        print("├── model.pkl")
        print("├── preprocessor.pkl")
        print("└── model_metadata.json")
        return True

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    prepare_deployment()
