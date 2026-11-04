import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from pathlib import Path

TRAIN_FEATURES_PATH = Path("data/processed/X_train_raw.csv")
TRAIN_LABELS_PATH = Path("data/processed/y_train.npy")
EVOLVED_LABELED_PATH = Path("data/monitoring/drift/evolved_labeled_data.csv")
UPDATED_DATASET_PATH = Path("data/processed/updated_training_data.csv")
SUMMARY_PATH = Path("outputs/retraining/updated_dataset_summary.csv")


def create_updated_dataset():
    print("[INFO] Combining baseline training data with drifted batch...")

    X_train = pd.read_csv(TRAIN_FEATURES_PATH)
    y_train = np.load(TRAIN_LABELS_PATH)
    X_train["churn"] = y_train

    evolved_df = pd.read_csv(EVOLVED_LABELED_PATH)

    combined_df = pd.concat([X_train, evolved_df], ignore_index=True)
    UPDATED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(UPDATED_DATASET_PATH, index=False)

    summary = {
        "baseline_train_rows": len(X_train),
        "new_drift_rows": len(evolved_df),
        "total_updated_rows": len(combined_df),
        "churn_rate": round(float(combined_df["churn"].mean()), 4),
        "output_path": str(UPDATED_DATASET_PATH)
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(SUMMARY_PATH, index=False)

    print(f"[SUCCESS] Updated training dataset created with {len(combined_df)} rows.")
    print(f"[SUCCESS] Saved to {UPDATED_DATASET_PATH}")
    return True


if __name__ == "__main__":
    create_updated_dataset()
