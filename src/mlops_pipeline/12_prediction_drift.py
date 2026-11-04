import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp
import yaml

from src.mlops_pipeline.inference_helper import clean_dataframe_for_inference

BASELINE_PATH = Path("data/processed/X_test_raw.csv")
EVOLVED_PATH = Path("data/monitoring/drift/evolved_data.csv")
REPORT_PATH = Path("outputs/drift/prediction_drift_report.csv")
KS_THRESHOLD = 0.10


def analyze_prediction_drift():
    print("=" * 70)
    print("PREDICTION DRIFT ANALYSIS")
    print("=" * 70)

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_path = Path("deployment/model.pkl") if Path("deployment/model.pkl").exists() else Path(config["paths"]["model"])
    preprocessor_path = Path("deployment/preprocessor.pkl") if Path("deployment/preprocessor.pkl").exists() else Path(config["paths"]["preprocessor"])

    if not model_path.exists() or not preprocessor_path.exists():
        model_path = Path("models/best_google_play_rating_model.pkl")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    baseline_df = pd.read_csv(BASELINE_PATH)
    evolved_df = pd.read_csv(EVOLVED_PATH)

    base_clean = clean_dataframe_for_inference(baseline_df)
    evolved_clean = clean_dataframe_for_inference(evolved_df)

    base_trans = preprocessor.transform(base_clean)
    evolved_trans = preprocessor.transform(evolved_clean)

    base_preds = model.predict(base_trans)
    base_probs = model.predict_proba(base_trans)[:, 1]

    evolved_preds = model.predict(evolved_trans)
    evolved_probs = model.predict_proba(evolved_trans)[:, 1]

    pred_ks, pred_p = ks_2samp(base_preds, evolved_preds)
    prob_ks, prob_p = ks_2samp(base_probs, evolved_probs)

    pred_status = "DRIFT" if pred_ks >= KS_THRESHOLD else "OK"
    prob_status = "DRIFT" if prob_ks >= KS_THRESHOLD else "OK"

    result = {
        "prediction_distribution_ks": float(pred_ks),
        "prediction_distribution_p_value": float(pred_p),
        "prediction_distribution_status": pred_status,
        "probability_distribution_ks": float(prob_ks),
        "probability_distribution_p_value": float(prob_p),
        "probability_distribution_status": prob_status,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(REPORT_PATH, index=False)

    print(f"Baseline mean churn prediction : {base_preds.mean():.4f}")
    print(f"Evolved mean churn prediction  : {evolved_preds.mean():.4f}")
    print(f"Prediction KS-statistic        : {pred_ks:.6f} ({pred_status})")
    print(f"Probability KS-statistic       : {prob_ks:.6f} ({prob_status})")
    print(f"Report saved to                : {REPORT_PATH}")
    return result


if __name__ == "__main__":
    analyze_prediction_drift()
