import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from pathlib import Path

SOURCE_X = Path("data/monitoring/drift/evolved_data.csv")
SOURCE_Y = Path("data/processed/y_test.npy")
OUTPUT_PATH = Path("data/monitoring/drift/evolved_labeled_data.csv")
RANDOM_SEED = 42


def create_concept_drift_dataset():
    if not SOURCE_X.exists() or not SOURCE_Y.exists():
        print("[ERROR] Required evolved data or baseline labels not found.")
        return False

    rng = np.random.default_rng(RANDOM_SEED)
    df = pd.read_csv(SOURCE_X).copy()
    y_base = np.load(SOURCE_Y).copy()

    # Simulate concept drift: user expectations rise, so apps receive lower ratings
    # Flip ~15% of positive class (0 -> 1) due to harsher review standards
    flip_mask = (y_base == 0) & (rng.random(len(y_base)) < 0.18)
    y_evolved = y_base.copy()
    y_evolved[flip_mask] = 1

    df["churn"] = y_evolved

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print("EVOLVED LABELED DATASET CREATED (CONCEPT DRIFT)")
    print("=" * 60)
    print(f"Output : {OUTPUT_PATH}")
    print(f"Rows   : {len(df)}")
    print(f"Original Churn Ratio: {y_base.mean():.4f}")
    print(f"Evolved Churn Ratio : {y_evolved.mean():.4f}")
    return True


if __name__ == "__main__":
    create_concept_drift_dataset()
