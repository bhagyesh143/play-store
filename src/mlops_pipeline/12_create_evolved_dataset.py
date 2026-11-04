import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os
import numpy as np
import pandas as pd
from pathlib import Path

SOURCE_PATH = Path("data/processed/X_test_raw.csv")
OUTPUT_PATH = Path("data/monitoring/drift/evolved_data.csv")
RANDOM_SEED = 42


def create_evolved_dataset():
    if not SOURCE_PATH.exists():
        print(f"[ERROR] Source test features not found: {SOURCE_PATH}")
        return False

    rng = np.random.default_rng(RANDOM_SEED)
    df = pd.read_csv(SOURCE_PATH).copy()

    # 1. Numerical drift: Reviews (inflated by viral distribution)
    if "Reviews" in df.columns:
        df["Reviews"] = (df["Reviews"] * rng.normal(1.35, 0.15, len(df))).clip(lower=0).round()

    # 2. Numerical drift: Size (larger modern applications)
    if "Size" in df.columns:
        df["Size"] = (df["Size"] * rng.normal(1.25, 0.10, len(df))).clip(lower=0.5, upper=150.0).round(2)

    # 3. Numerical drift: Installs (market expansion)
    if "Installs" in df.columns:
        df["Installs"] = (df["Installs"] * rng.choice([1.0, 1.5, 2.0, 5.0], size=len(df), p=[0.2, 0.4, 0.3, 0.1])).round()

    # 4. Numerical drift: Price
    if "Price" in df.columns:
        df["Price"] = (df["Price"] * rng.choice([1.0, 1.2, 1.5], size=len(df), p=[0.7, 0.2, 0.1])).round(2)

    # 5. Categorical drift: Category
    if "Category" in df.columns:
        top_cats = ["GAME", "FAMILY", "TOOLS", "PRODUCTIVITY", "FINANCE"]
        df["Category"] = rng.choice(top_cats, size=len(df), p=[0.40, 0.25, 0.15, 0.10, 0.10])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print("EVOLVED PLAY STORE DATASET CREATED")
    print("=" * 60)
    print(f"Source   : {SOURCE_PATH}")
    print(f"Output   : {OUTPUT_PATH}")
    print(f"Rows     : {len(df)}")
    print(f"Columns  : {len(df.columns)}")
    print("Modified features with simulated drift:")
    print("- Reviews (mean shift: +35%)")
    print("- Size (mean shift: +25%)")
    print("- Installs (categorical scale expansion)")
    print("- Category (shifted towards GAME and FAMILY)")
    return True


if __name__ == "__main__":
    create_evolved_dataset()
