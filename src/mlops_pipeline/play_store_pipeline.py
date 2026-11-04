import json
import os
import subprocess
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

STAGES = [
    ("Data Loading & Lineage", "src/mlops_pipeline/01_load_data.py"),
    ("Raw Data Validation", "src/mlops_pipeline/02_validate_data.py"),
    ("Preprocessing & Featurization", "src/mlops_pipeline/03_preprocessing.py"),
    ("Processed Output Validation", "src/mlops_pipeline/04_validate_outputs.py"),
    ("Training + MLflow Tracking + Registry", "src/mlops_pipeline/05_train_registry.py"),
    ("Model Evaluation", "src/mlops_pipeline/06_model_evaluation.py"),
    ("Model Quality Gate", "src/mlops_pipeline/07_quality_gate.py"),
    ("Automated Model Lifecycle", "src/mlops_pipeline/08_automate_lifecycle.py"),
    ("Conditional Deployment Rollout", "src/mlops_pipeline/deploy_if_promoted.py"),
    ("Production Inference Demo", "src/mlops_pipeline/09_inference.py"),
]


def run_pipeline():
    print("[INFO] ==================================================")
    print("[INFO] Starting Google Play Store Rating MLOps Pipeline")
    print("[INFO] ==================================================")

    run_log = {"pipeline": "Google Play Store Rating MLOps", "stages": []}

    for stage_name, script_path in STAGES:
        print("\n" + "=" * 70)
        print(f"RUNNING: {stage_name}")
        print("=" * 70)

        result = subprocess.run([sys.executable, script_path])
        status = "PASSED" if result.returncode == 0 else "FAILED"

        run_log["stages"].append({
            "stage": stage_name,
            "script": script_path,
            "status": status,
            "exit_code": result.returncode,
        })

        print(f"--> {stage_name}: {status}")

        if result.returncode != 0:
            print("[HALT] Pipeline stopped because this stage failed.")
            break

    all_passed = len(run_log["stages"]) == len(STAGES) and all(
        s["status"] == "PASSED" for s in run_log["stages"]
    )
    run_log["overall_status"] = "PASSED" if all_passed else "FAILED"

    os.makedirs("logs", exist_ok=True)
    log_path = "logs/play_store_pipeline_run_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=4)

    print("\n" + "=" * 70)
    if all_passed:
        print("[SUCCESS] Google Play Store MLOps Pipeline completed successfully.")
    else:
        print("[ERROR] Google Play Store MLOps Pipeline failed.")
    print(f"[INFO] Run log: {log_path}")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    passed = run_pipeline()
    sys.exit(0 if passed else 1)
