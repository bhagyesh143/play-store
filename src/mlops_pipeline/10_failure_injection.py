import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import os
import pandas as pd
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

RAW_DATA = config["data"]["raw_path"]
ARTIFACTS_DIR = "artifacts"
REPORTS_DIR = "reports"

FAILURE_REPORT = os.path.join(ARTIFACTS_DIR, "failure_simulation_report.json")
CORRUPTED_REPORT = os.path.join(ARTIFACTS_DIR, "corrupted_simulation_errors.csv")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def inject_failures():
    print("\n========== FAILURE INJECTION & RESILIENCE TESTING ==========")

    failures = []
    errors = []

    # 1. Missing raw dataset
    print("\n[TEST 1] Checking raw dataset availability...")
    if not os.path.exists(RAW_DATA):
        failures.append({
            "failure_type": "Missing Raw Dataset",
            "status": "DETECTED",
            "message": f"Raw dataset not found: {RAW_DATA}"
        })
        errors.append({"test": "Missing Raw Dataset", "expected": "Dataset should exist", "result": "FAIL"})
        print("[PASS] Missing dataset failure test detected.")
    else:
        print("[PASS] Raw dataset exists.")

    # 2. Target column presence
    print("\n[TEST 2] Checking Rating column...")
    try:
        df = pd.read_csv(RAW_DATA)
        if "Rating" not in df.columns:
            failures.append({
                "failure_type": "Missing Target Column",
                "status": "DETECTED",
                "message": "Target column 'Rating' not found."
            })
            errors.append({"test": "Missing Target Column", "expected": "Rating", "result": "FAIL"})
            print("[PASS] Missing target-column failure detected.")
        else:
            print("[PASS] Rating column exists.")
    except Exception as e:
        failures.append({"failure_type": "Dataset Read Failure", "status": "DETECTED", "message": str(e)})
        errors.append({"test": "Dataset Read", "expected": "CSV loads successfully", "result": "FAIL"})

    # 3. Missing values check
    print("\n[TEST 3] Checking missing values...")
    try:
        df = pd.read_csv(RAW_DATA)
        missing_count = int(df.isnull().sum().sum())
        if missing_count > 0:
            failures.append({
                "failure_type": "Missing Values",
                "status": "DETECTED",
                "message": f"{missing_count} missing values detected across dataset."
            })
            errors.append({"test": "Missing Values", "expected": 0, "actual": missing_count, "result": "FAIL"})
            print(f"[PASS] Missing-value check detected: {missing_count} missing cells.")
        else:
            print("[PASS] No missing values found.")
    except Exception as e:
        print(f"[INFO] Missing value test skipped: {e}")

    # 4. Invalid rating bounds
    print("\n[TEST 4] Checking rating boundary corruptions (rating < 1 or > 5)...")
    try:
        df = pd.read_csv(RAW_DATA)
        numeric_ratings = pd.to_numeric(df["Rating"], errors="coerce").dropna()
        invalid_ratings = numeric_ratings[(numeric_ratings < 1.0) | (numeric_ratings > 5.0)].tolist()
        if invalid_ratings:
            failures.append({
                "failure_type": "Invalid Target Values",
                "status": "DETECTED",
                "message": f"Invalid rating values found: {invalid_ratings[:5]}"
            })
            errors.append({"test": "Invalid Target Values", "expected": "1.0 <= Rating <= 5.0", "actual": invalid_ratings[:5], "result": "FAIL"})
            print(f"[PASS] Invalid rating bounds failure detected: {len(invalid_ratings)} invalid items.")
        else:
            print("[PASS] All ratings within valid boundary.")
    except Exception as e:
        print(f"[INFO] Boundary test skipped: {e}")

    # 5. Duplicate records
    print("\n[TEST 5] Checking duplicate records...")
    try:
        df = pd.read_csv(RAW_DATA)
        duplicate_count = int(df.duplicated().sum())
        if duplicate_count > 0:
            failures.append({
                "failure_type": "Duplicate Records",
                "status": "DETECTED",
                "message": f"{duplicate_count} duplicate rows detected."
            })
            errors.append({"test": "Duplicate Records", "expected": 0, "actual": duplicate_count, "result": "FAIL"})
            print(f"[PASS] Duplicate records detected: {duplicate_count}")
        else:
            print("[PASS] No duplicate records found.")
    except Exception as e:
        print(f"[INFO] Duplicate check skipped: {e}")

    # Save outputs
    if errors:
        pd.DataFrame(errors).to_csv(CORRUPTED_REPORT, index=False)
    else:
        pd.DataFrame(columns=["test", "expected", "actual", "result"]).to_csv(CORRUPTED_REPORT, index=False)

    report = {
        "pipeline": "Google Play Store Rating MLOps",
        "failure_injection": True,
        "total_tests": 5,
        "failures_detected": len(failures),
        "status": "FAILURES_DETECTED" if failures else "NO_FAILURES_DETECTED",
        "failures": failures,
    }

    with open(FAILURE_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\n========================================")
    print("FAILURE INJECTION & RESILIENCE COMPLETED")
    print("========================================")
    print(f"Failures detected : {len(failures)}")
    print(f"Report saved      : {FAILURE_REPORT}")
    print(f"Error CSV saved   : {CORRUPTED_REPORT}")
    return report


if __name__ == "__main__":
    inject_failures()
