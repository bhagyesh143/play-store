import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
import yaml


def validate_outputs():
    print("[INFO] Starting Processed Output Validation...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    processed_dir = config["data"]["processed_dir"]
    preprocessor_path = config["paths"]["preprocessor"]

    required_files = [
        os.path.join(processed_dir, "X_train_final.npy"),
        os.path.join(processed_dir, "X_test_final.npy"),
        os.path.join(processed_dir, "y_train.npy"),
        os.path.join(processed_dir, "y_test.npy"),
        preprocessor_path,
    ]

    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"[ERROR] Missing required preprocessed output files: {missing}")
        return False

    X_train = np.load(os.path.join(processed_dir, "X_train_final.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test_final.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))

    print(f"[INFO] X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"[INFO] X_test shape : {X_test.shape}, y_test shape : {y_test.shape}")

    # Check sample count alignment
    if X_train.shape[0] != y_train.shape[0]:
        print("[ERROR] Mismatch between X_train and y_train row counts!")
        return False

    if X_test.shape[0] != y_test.shape[0]:
        print("[ERROR] Mismatch between X_test and y_test row counts!")
        return False

    # Check feature dimension alignment
    if X_train.shape[1] != X_test.shape[1]:
        print("[ERROR] Feature dimension mismatch between train and test sets!")
        return False

    # Check NaN values
    train_nans = int(np.isnan(X_train).sum())
    test_nans = int(np.isnan(X_test).sum())
    if train_nans > 0 or test_nans > 0:
        print(f"[ERROR] NaN values found: train={train_nans}, test={test_nans}")
        return False

    # Check preprocessor loading
    try:
        preprocessor = joblib.load(preprocessor_path)
        print("[SUCCESS] Preprocessor loaded and verified.")
    except Exception as e:
        print(f"[ERROR] Failed to load preprocessor: {e}")
        return False

    os.makedirs("artifacts", exist_ok=True)
    report = {
        "validation_status": "PASSED",
        "X_train_shape": list(X_train.shape),
        "X_test_shape": list(X_test.shape),
        "y_train_shape": list(y_train.shape),
        "y_test_shape": list(y_test.shape),
        "train_nans": train_nans,
        "test_nans": test_nans,
        "features_count": int(X_train.shape[1])
    }
    pd.DataFrame([report]).to_csv("artifacts/output_validation_report.csv", index=False)
    print("[SUCCESS] Processed Output Validation completed successfully. Report saved to artifacts/output_validation_report.csv")
    return True


if __name__ == "__main__":
    passed = validate_outputs()
    sys.exit(0 if passed else 1)
