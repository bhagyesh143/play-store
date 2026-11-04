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
import yaml
from mlflow.tracking import MlflowClient

V2_MODEL_PATH = Path("models/retrained_model_v2.pkl")
V2_PREPROCESSOR_PATH = Path("models/retrained_preprocessor_v2.pkl")
COMPARISON_REPORT_PATH = Path("outputs/retraining/model_comparison_report.json")
REGISTRY_REPORT_PATH = Path("outputs/retraining/mlflow_registry_update_report.json")


def update_registry():
    print("[INFO] Updating MLflow Model Registry for Retrained Candidate...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"]["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    model_name = config["registry"]["model_name"]

    with open(COMPARISON_REPORT_PATH, "r", encoding="utf-8") as f:
        comp_report = json.load(f)

    champion = comp_report.get("champion", "production_v1")
    model = joblib.load(V2_MODEL_PATH)

    with mlflow.start_run(run_name="Register_Retrained_v2") as run:
        mlflow.log_artifact(str(V2_PREPROCESSOR_PATH), artifact_path="preprocessing_pipeline_v2")
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )

    # Get newly created version
    versions = client.search_model_versions(f"name='{model_name}'")
    latest_v2 = max(versions, key=lambda v: int(v.version))

    if champion == "candidate_v2":
        print(f"[INFO] Candidate v2 is champion! Transitioning Version {latest_v2.version} to Production...")
        client.transition_model_version_stage(
            name=model_name,
            version=latest_v2.version,
            stage="Production",
            archive_existing_versions=True
        )
        status_note = f"Version {latest_v2.version} promoted to Production."
    else:
        print(f"[INFO] Candidate v2 archived.")
        client.transition_model_version_stage(
            name=model_name,
            version=latest_v2.version,
            stage="Archived",
            archive_existing_versions=False
        )
        status_note = f"Version {latest_v2.version} archived (existing Production maintained)."

    report = {
        "model_name": model_name,
        "new_version": str(latest_v2.version),
        "stage": latest_v2.current_stage,
        "champion_status": champion,
        "decision": status_note,
        "run_id": run.info.run_id
    }

    REGISTRY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"[SUCCESS] Registry update complete: {status_note}")
    return True


if __name__ == "__main__":
    update_registry()
