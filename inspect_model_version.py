import sys
import yaml
import mlflow
from mlflow import MlflowClient

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
client = MlflowClient()

model_name = config["registry"]["model_name"]
version = sys.argv[1] if len(sys.argv) > 1 else "1"

try:
    mv = client.get_model_version(model_name, version)
    print("=" * 60)
    print("MODEL NAME        :", mv.name)
    print("VERSION           :", mv.version)
    print("STAGE             :", mv.current_stage)
    print("RUN ID            :", mv.run_id)
    print("SOURCE            :", mv.source)
    print("STATUS            :", mv.status)
    print("CREATION TIMESTAMP:", mv.creation_timestamp)
except Exception as e:
    print(f"[ERROR] Could not inspect version {version} of model {model_name}: {e}")
