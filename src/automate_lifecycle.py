
import mlflow
from mlflow import MlflowClient


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "Google_Play_Store_Random_Forest"


print("[INFO] =========================================")
print("[INFO] Lab 6 - Model Lifecycle Management")
print("[INFO] =========================================")


# ============================================================
# CREATE CLIENT
# ============================================================

client = MlflowClient()


# ============================================================
# GET REGISTERED MODEL
# ============================================================

print("[INFO] Searching for registered model...")

try:

    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

except Exception as e:

    print(
        "[ERROR] Could not find registered model."
    )

    print(e)

    raise SystemExit(1)


if len(versions) == 0:

    print(
        "[ERROR] No model versions found."
    )

    raise SystemExit(1)


print(
    f"[SUCCESS] Found {len(versions)} model version(s)."
)


# ============================================================
# DISPLAY VERSIONS
# ============================================================

print("\n[INFO] Model Versions:")

for version in versions:

    print(
        f"Version: {version.version}"
    )

    print(
        f"Status: {version.status}"
    )

    print(
        f"Run ID: {version.run_id}"
    )

    print(
        "-----------------------------"
    )


# ============================================================
# FIND LATEST VERSION
# ============================================================

latest_version = max(
    versions,
    key=lambda x: int(x.version)
)

version_number = latest_version.version


print(
    f"[INFO] Latest version: {version_number}"
)


# ============================================================
# ADD DESCRIPTION
# ============================================================

try:

    client.update_model_version(
        name=MODEL_NAME,
        version=version_number,
        description=(
            "Random Forest model trained on "
            "Google Play Store dataset. "
            "Target is a rating-based proxy."
        )
    )

    print(
        "[SUCCESS] Model description updated."
    )

except Exception as e:

    print(
        "[WARNING] Could not update description:"
    )

    print(e)


# ============================================================
# SET MODEL ALIAS
# ============================================================

try:

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=version_number
    )

    print(
        f"[SUCCESS] Version {version_number} "
        "assigned alias: champion"
    )

except Exception as e:

    print(
        "[WARNING] Could not assign alias:"
    )

    print(e)


# ============================================================
# COMPLETE
# ============================================================

print("\n[INFO] =========================================")

print(
    "[SUCCESS] Model lifecycle automation completed!"
)

print(
    f"[INFO] Model: {MODEL_NAME}"
)

print(
    f"[INFO] Champion Version: {version_number}"
)

print(
    "[INFO] =========================================")
