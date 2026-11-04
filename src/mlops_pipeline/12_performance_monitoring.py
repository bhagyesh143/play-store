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

X_TEST_PATH = Path("data/processed/X_test_raw.csv")
Y_TEST_PATH = Path("data/processed/y_test.npy")


def main():
    print("=" * 60)
    print("LIVE PRODUCTION MODEL PERFORMANCE MONITORING")
    print("=" * 60)

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_path = Path("deployment/model.pkl") if Path("deployment/model.pkl").exists() else Path(config["paths"]["model"])
    preprocessor_path = Path("deployment/preprocessor.pkl") if Path("deployment/preprocessor.pkl").exists() else Path(config["paths"]["preprocessor"])

    if not model_path.exists():
        model_path = Path("models/best_google_play_rating_model.pkl")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    X_test = pd.read_csv(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    X_clean = clean_dataframe_for_inference(X_test)
    X_trans = preprocessor.transform(X_clean)

    preds = model.predict(X_trans)
    probs = model.predict_proba(X_trans)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1_score": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
    }

    for k, v in metrics.items():
        print(f"  {k.replace('_', ' ').title():15}: {v:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
