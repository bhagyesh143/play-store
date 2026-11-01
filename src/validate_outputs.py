import os
import numpy as np
import pandas as pd
import joblib


print("[INFO] =========================================")
print("[INFO] Lab 5 - Output Validation")
print("[INFO] =========================================")


# ============================================================
# 1. REQUIRED OUTPUTS
# ============================================================

required_files = [
    "models/preprocessor.pkl",
    "artifacts/X_train.npy",
    "artifacts/X_test.npy",
    "artifacts/y_train.npy",
    "artifacts/y_test.npy",
    "artifacts/preprocessing_summary.json",
    "artifacts/data_validation_report.csv"
]


# ============================================================
# 2. CHECK FILES
# ============================================================

print("[INFO] Checking required output files...")

missing_files = []

for file_path in required_files:

    if os.path.exists(file_path):

        print(
            f"[OK] {file_path}"
        )

    else:

        print(
            f"[MISSING] {file_path}"
        )

        missing_files.append(
            file_path
        )


if missing_files:

    print("\n[ERROR] Required output files are missing.")

    for file_path in missing_files:
        print(
            "[ERROR]",
            file_path
        )

    raise SystemExit(1)


# ============================================================
# 3. LOAD OUTPUTS
# ============================================================

print("\n[INFO] Loading processed outputs...")

X_train = np.load(
    "artifacts/X_train.npy"
)

X_test = np.load(
    "artifacts/X_test.npy"
)

y_train = np.load(
    "artifacts/y_train.npy"
)

y_test = np.load(
    "artifacts/y_test.npy"
)


# ============================================================
# 4. CHECK SHAPES
# ============================================================

print("\n[INFO] Output shapes:")

print(
    "X_train:",
    X_train.shape
)

print(
    "X_test:",
    X_test.shape
)

print(
    "y_train:",
    y_train.shape
)

print(
    "y_test:",
    y_test.shape
)


# ============================================================
# 5. CHECK FOR NaN
# ============================================================

print("\n[INFO] Checking NaN values...")

if np.isnan(X_train).any():

    print("[ERROR] X_train contains NaN values.")
    raise SystemExit(1)

if np.isnan(X_test).any():

    print("[ERROR] X_test contains NaN values.")
    raise SystemExit(1)

if np.isnan(y_train).any():

    print("[ERROR] y_train contains NaN values.")
    raise SystemExit(1)

if np.isnan(y_test).any():

    print("[ERROR] y_test contains NaN values.")
    raise SystemExit(1)

print(
    "[SUCCESS] No NaN values found."
)


# ============================================================
# 6. CHECK PREPROCESSOR
# ============================================================

print("\n[INFO] Checking saved preprocessor...")

preprocessor = joblib.load(
    "models/preprocessor.pkl"
)

print(
    "[SUCCESS] Preprocessor loaded successfully."
)


# ============================================================
# 7. CHECK TRAIN / TEST SIZE
# ============================================================

if len(X_train) != len(y_train):

    print(
        "[ERROR] X_train and y_train sizes do not match."
    )

    raise SystemExit(1)


if len(X_test) != len(y_test):

    print(
        "[ERROR] X_test and y_test sizes do not match."
    )

    raise SystemExit(1)


print(
    "[SUCCESS] Training and testing sizes are valid."
)


# ============================================================
# 8. CREATE VALIDATION REPORT
# ============================================================

report = {
    "X_train_rows": int(X_train.shape[0]),
    "X_train_columns": int(X_train.shape[1]),
    "X_test_rows": int(X_test.shape[0]),
    "X_test_columns": int(X_test.shape[1]),
    "y_train_rows": int(y_train.shape[0]),
    "y_test_rows": int(y_test.shape[0]),
    "nan_values": False,
    "preprocessor_available": True,
    "validation_status": "PASSED"
}


pd.DataFrame(
    [report]
).to_csv(
    "artifacts/output_validation_report.csv",
    index=False
)


print(
    "\n[SUCCESS] Output validation completed!"
)

print(
    "[SUCCESS] Lab 5 validation PASSED!"
)

print(
    "[INFO] Report saved to:"
)

print(
    "artifacts/output_validation_report.csv"
)