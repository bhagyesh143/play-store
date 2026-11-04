import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import os
import sys
import yaml
import mlflow
from mlflow.tracking import MlflowClient


def configure_mlflow(config):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def automate_model_lifecycle():
    print("[INFO] Starting Automated Model Lifecycle Manager...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    configure_mlflow(config)
    client = MlflowClient()
    model_name = config["registry"]["model_name"]
    metric_to_optimize = config["registry"].get("decision_metric", "f1_score")

    try:
        # 1. Find new models and move them to Staging
        all_versions = client.search_model_versions(f"name='{model_name}'")
        new_models = [mv for mv in all_versions if mv.current_stage == "None"]

        for mv in new_models:
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage="Staging",
                archive_existing_versions=False,
            )
            print(f"[INFO] Version {mv.version} moved to Staging.")

        # 2. Find Staging models
        all_versions = client.search_model_versions(f"name='{model_name}'")
        models_staging = [mv for mv in all_versions if mv.current_stage == "Staging"]

        if not models_staging:
            print("[INFO] No models in Staging. Nothing to promote.")
            return True

        # 3. Select best Staging model
        best_model = None
        best_score = -1.0

        for mv in models_staging:
            run = client.get_run(mv.run_id)
            score = run.data.metrics.get(metric_to_optimize, 0.0)
            print(f"[INFO] Staging Version {mv.version} | {metric_to_optimize}={score:.4f}")
            if score > best_score:
                best_score = score
                best_model = mv

        # 4. Find current Production model
        production_models = [mv for mv in all_versions if mv.current_stage == "Production"]
        production_model = production_models[0] if production_models else None

        if production_model is None:
            move_to_production = True
            production_score = 0.0
            print("[INFO] No Production model currently exists. Staging model will become Production.")
        else:
            production_run = client.get_run(production_model.run_id)
            production_score = production_run.data.metrics.get(metric_to_optimize, 0.0)
            print(f"[INFO] Current Production Version {production_model.version} | {metric_to_optimize}={production_score:.4f}")
            move_to_production = best_score > production_score

        # 5. Promote or archive
        if move_to_production:
            client.transition_model_version_stage(
                name=model_name,
                version=best_model.version,
                stage="Production",
                archive_existing_versions=False,
            )
            if production_model:
                client.transition_model_version_stage(
                    name=model_name,
                    version=production_model.version,
                    stage="Archived",
                    archive_existing_versions=False,
                )
            print(f"[SUCCESS] Version {best_model.version} promoted to Production.")
            current_prod_version = best_model.version
        else:
            client.transition_model_version_stage(
                name=model_name,
                version=best_model.version,
                stage="Archived",
                archive_existing_versions=False,
            )
            print(f"[INFO] Version {best_model.version} archived (did not outperform Production).")
            current_prod_version = production_model.version if production_model else None

        # 6. Archive remaining Staging models
        final_versions = client.search_model_versions(f"name='{model_name}'")
        for mv in final_versions:
            if mv.current_stage == "Staging":
                client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage="Archived",
                    archive_existing_versions=False,
                )

        os.makedirs("reports", exist_ok=True)
        lifecycle_report = {
            "stage": "Automated Model Lifecycle Manager",
            "model_name": model_name,
            "decision_metric": metric_to_optimize,
            "candidate_version": best_model.version if best_model else None,
            "candidate_score": float(best_score),
            "production_version_before": production_model.version if production_model else None,
            "production_score_before": float(production_score),
            "promoted": move_to_production,
            "production_version_after": current_prod_version,
        }

        with open(config["paths"]["lifecycle_report"], "w", encoding="utf-8") as f:
            json.dump(lifecycle_report, f, indent=4)

        print(f"[SUCCESS] Lifecycle report saved to {config['paths']['lifecycle_report']}")
        return True

    except Exception as exc:
        print(f"[ERROR] Failed to automate model lifecycle: {exc}")
        return False


if __name__ == "__main__":
    passed = automate_model_lifecycle()
    sys.exit(0 if passed else 1)
