import mlflow
from mlflow import MlflowClient
import yaml
import os

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
client = MlflowClient()
model_name = config["registry"]["model_name"]

print("=" * 60)
print("MLFLOW MODEL REGISTRY INSPECTION")
print("=" * 60)

try:
    model = client.get_registered_model(model_name)
    print("MODEL:", model.name)
    print("VERSIONS:")
    versions = client.search_model_versions(f"name='{model_name}'")
    for version in versions:
        run = client.get_run(version.run_id)
        f1 = run.data.metrics.get("f1_score", "N/A")
        acc = run.data.metrics.get("accuracy", "N/A")
        print(
            f"  version={version.version:<3} "
            f"stage={version.current_stage:<12} "
            f"run_id={version.run_id} "
            f"f1={f1} acc={acc}"
        )
except Exception as e:
    print(f"[INFO] Registered model '{model_name}' query notice: {e}")
