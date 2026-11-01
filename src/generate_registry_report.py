
import os
import pandas as pd
from mlflow import MlflowClient


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "Google_Play_Store_Random_Forest"

REPORT_PATH = (
    "artifacts/model_registry_report.csv"
)

os.makedirs(
    "artifacts",
    exist_ok=True
)


print("[INFO] =========================================")
print("[INFO] Lab 6 - Model Registry Report")
print("[INFO] =========================================")


# ============================================================
# CREATE MLFLOW CLIENT
# ============================================================

client = MlflowClient()


# ============================================================
# GET MODEL VERSIONS
# ============================================================

print(
    "[INFO] Reading registered model information..."
)

try:

    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

except Exception as e:

    print(
        "[ERROR] Could not read model registry."
    )

    print(e)

    raise SystemExit(1)


if len(versions) == 0:

    print(
        "[ERROR] No registered model versions found."
    )

    raise SystemExit(1)


# ============================================================
# CREATE REPORT
# ============================================================

records = []


for version in versions:

    records.append(
        {
            "model_name": MODEL_NAME,
            "version": version.version,
            "run_id": version.run_id,
            "status": version.status,
            "source": version.source,
            "run_link": version.run_link
        }
    )


report = pd.DataFrame(
    records
)


# ============================================================
# SORT VERSIONS
# ============================================================

report["version"] = report[
    "version"
].astype(int)

report = report.sort_values(
    "version"
)


# ============================================================
# SAVE REPORT
# ============================================================

report.to_csv(
    REPORT_PATH,
    index=False
)


# ============================================================
# DISPLAY REPORT
# ============================================================

print("\n[INFO] Registered Models:")

print(
    report.to_string(
        index=False
    )
)


print(
    "\n[SUCCESS] Registry report created!"
)

print(
    "[INFO] Report saved to:",
    REPORT_PATH
)


print(
    "\n[INFO] ========================================="
)

print(
    "[SUCCESS] Lab 6 completed!"
)

print(
    "[INFO] ========================================="
)
