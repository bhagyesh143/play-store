import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os
import yaml
import mlflow
from mlflow.tracking import MlflowClient

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
client = MlflowClient()
model_name = config["registry"]["model_name"]

print("=" * 60)
print(f"CHECKING MLFLOW REGISTRY: {model_name}")
print("=" * 60)

versions = client.search_model_versions(f"name='{model_name}'")
for v in versions:
    run = client.get_run(v.run_id)
    f1 = run.data.metrics.get("f1_score", "N/A")
    acc = run.data.metrics.get("accuracy", "N/A")
    print(f"Version {v.version:<3} | Stage: {v.current_stage:<12} | Run: {v.run_id} | F1: {f1} | Acc: {acc}")
print("=" * 60)
