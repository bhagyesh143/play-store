import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import yaml

from src.mlops_pipeline.inference_helper import clean_dataframe_for_inference

TEST_FEATURES_PATH = Path("data/processed/X_test_raw.csv")
TEST_LABELS_PATH = Path("data/processed/y_test.npy")

V1_MODEL_PATH = Path("models/random_forest_model.pkl")
V1_PREPROCESSOR_PATH = Path("models/preprocessor.pkl")

V2_MODEL_PATH = Path("models/retrained_model_v2.pkl")
V2_PREPROCESSOR_PATH = Path("models/retrained_preprocessor_v2.pkl")

REPORT_PATH = Path("outputs/retraining/model_comparison_report.csv")
JSON_REPORT_PATH = Path("outputs/retraining/model_comparison_report.json")


def evaluate_pipeline(model_path, preprocessor_path, X_test, y_test):
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    X_clean = clean_dataframe_for_inference(X_test)
    X_trans = preprocessor.transform(X_clean)

    preds = model.predict(X_trans)
    probs = model.predict_proba(X_trans)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1_score": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
    }


def compare_models():
    print("=" * 70)
    print("MODEL COMPARISON (PRODUCTION V1 VS RETRAINED V2)")
    print("=" * 70)

    X_test = pd.read_csv(TEST_FEATURES_PATH)
    y_test = np.load(TEST_LABELS_PATH)

    v1_metrics = evaluate_pipeline(V1_MODEL_PATH, V1_PREPROCESSOR_PATH, X_test, y_test)
    v2_metrics = evaluate_pipeline(V2_MODEL_PATH, V2_PREPROCESSOR_PATH, X_test, y_test)

    rows = []
    for m in v1_metrics:
        b_val = v1_metrics[m]
        c_val = v2_metrics[m]
        rows.append({
            "metric": m,
            "production_v1": round(b_val, 4),
            "candidate_v2": round(c_val, 4),
            "improvement": round(c_val - b_val, 4),
            "improved": c_val > b_val,
        })

    report_df = pd.DataFrame(rows)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(REPORT_PATH, index=False)

    champion = "candidate_v2" if v2_metrics["f1_score"] >= v1_metrics["f1_score"] else "production_v1"

    summary_json = {
        "production_v1": v1_metrics,
        "candidate_v2": v2_metrics,
        "champion": champion,
        "comparison_table": rows,
    }
    with open(JSON_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=4)

    print(report_df.to_string(index=False))
    print(f"\n[INFO] Decision Champion: {champion.upper()}")
    print(f"[SUCCESS] Reports saved: {REPORT_PATH} and {JSON_REPORT_PATH}")
    return summary_json


if __name__ == "__main__":
    compare_models()
