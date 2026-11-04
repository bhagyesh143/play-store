import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import os
import sys
import yaml


def run_quality_gate():
    print("[INFO] Starting Model Quality Gate...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    report_path = config["paths"]["evaluation_report"]
    metric_name = config["quality_gate"]["metric"]
    minimum_value = float(config["quality_gate"]["minimum_value"])

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Evaluation report not found: {report_path}")
        return False

    actual_value = float(report["metrics"].get(metric_name, 0.0))
    passed = actual_value >= minimum_value

    print(
        f"[INFO] Quality Gate Evaluation: {metric_name} = {actual_value:.4f} | "
        f"required threshold >= {minimum_value:.4f}"
    )

    os.makedirs(os.path.dirname(config["paths"]["quality_gate_report"]), exist_ok=True)

    gate_report = {
        "stage": "Lab 8 - Model Quality Gate",
        "metric": metric_name,
        "minimum_value": minimum_value,
        "actual_value": actual_value,
        "status": "PASSED" if passed else "FAILED",
    }

    with open(config["paths"]["quality_gate_report"], "w", encoding="utf-8") as f:
        json.dump(gate_report, f, indent=4)

    if passed:
        print(f"[SUCCESS] Model Quality Gate PASSED ({metric_name} >= {minimum_value}). Model eligible for lifecycle promotion.")
    else:
        print(f"[ERROR] Model Quality Gate FAILED ({metric_name} < {minimum_value}). Model will be blocked from production promotion.")

    return passed


if __name__ == "__main__":
    passed = run_quality_gate()
    sys.exit(0 if passed else 1)
