import sys
import yaml
import mlflow
from mlflow import MlflowClient

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
client = MlflowClient()

run_id = sys.argv[1] if len(sys.argv) > 1 else None

if not run_id:
    # Get latest run
    experiment = client.get_experiment_by_name(config["mlflow"]["experiment_name"])
    if experiment:
        runs = client.search_runs(experiment_ids=[experiment.experiment_id], max_results=1)
        if runs:
            run_id = runs[0].info.run_id

if not run_id:
    print("[ERROR] No run ID provided and no runs found.")
    sys.exit(1)

run = client.get_run(run_id)

print("=" * 60)
print("RUN ID       :", run.info.run_id)
print("EXPERIMENT ID:", run.info.experiment_id)
print("STATUS       :", run.info.status)
print("ARTIFACT URI :", run.info.artifact_uri)
print("\nMETRICS:")
for k, v in run.data.metrics.items():
    print(f"  {k}: {v}")
print("\nPARAMETERS:")
for k, v in run.data.params.items():
    print(f"  {k}: {v}")

print("\nARTIFACTS:")
artifacts = client.list_artifacts(run_id)
for artifact in artifacts:
    print(f"  {artifact.path} | is_dir={artifact.is_dir}")
    if artifact.is_dir:
        children = client.list_artifacts(run_id, artifact.path)
        for child in children:
            print(f"    └── {child.path} | is_dir={child.is_dir}")
