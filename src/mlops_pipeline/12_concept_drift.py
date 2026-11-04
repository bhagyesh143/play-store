import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import yaml

from src.mlops_pipeline.inference_helper import clean_dataframe_for_inference

BASELINE_X_PATH = Path("data/processed/X_test_raw.csv")
BASELINE_Y_PATH = Path("data/processed/y_test.npy")
EVOLVED_PATH = Path("data/monitoring/drift/evolved_labeled_data.csv")
REPORT_PATH = Path("outputs/drift/concept_drift_report.csv")


def evaluate(model, preprocessor, X, y):
    X_clean = clean_dataframe_for_inference(X)
    X_trans = preprocessor.transform(X_clean)
    preds = model.predict(X_trans)
    probs = model.predict_proba(X_trans)[:, 1]

    return {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probs)),
    }


def analyze_concept_drift():
    print("=" * 70)
    print("CONCEPT DRIFT & PERFORMANCE DEGRADATION ANALYSIS")
    print("=" * 70)

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_path = Path("deployment/model.pkl") if Path("deployment/model.pkl").exists() else Path(config["paths"]["model"])
    preprocessor_path = Path("deployment/preprocessor.pkl") if Path("deployment/preprocessor.pkl").exists() else Path(config["paths"]["preprocessor"])

    if not model_path.exists():
        model_path = Path("models/best_google_play_rating_model.pkl")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    base_X = pd.read_csv(BASELINE_X_PATH)
    base_y = np.load(BASELINE_Y_PATH)

    evolved_df = pd.read_csv(EVOLVED_PATH)
    target_col = config["data"].get("target_column", "churn")
    evolved_X = evolved_df.drop(columns=[target_col])
    evolved_y = evolved_df[target_col].to_numpy()

    base_metrics = evaluate(model, preprocessor, base_X, base_y)
    evolved_metrics = evaluate(model, preprocessor, evolved_X, evolved_y)

    results = []
    for metric in base_metrics:
        b_val = base_metrics[metric]
        e_val = evolved_metrics[metric]
        change = e_val - b_val
        results.append({
            "metric": metric,
            "baseline": round(b_val, 4),
            "evolved": round(e_val, 4),
            "change": round(change, 4),
            "status": "DRIFT" if change < -0.05 else "OK",
        })

    report = pd.DataFrame(results)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)

    print("\nPERFORMANCE DEGRADATION AUDIT:")
    print(report.to_string(index=False))
    print(f"\nReport saved to: {REPORT_PATH}")
    return report


if __name__ == "__main__":
    analyze_concept_drift()
