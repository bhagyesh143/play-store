import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
from pathlib import Path
import numpy as np
import pandas as pd

LOG_PATH = Path("logs/inference/prediction_logs.jsonl")


def main():
    print("=" * 60)
    print("REAL-TIME PREDICTION AUDIT LOG MONITORING")
    print("=" * 60)

    if not LOG_PATH.exists():
        print(f"[INFO] Inference log file not found at {LOG_PATH}")
        return

    records = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except:
                    pass

    if not records:
        print("[INFO] Log file is empty.")
        return

    df = pd.DataFrame(records)
    print(f"Total Requests Logged: {len(df)}")
    if "predicted_rating" in df.columns:
        valid_ratings = df["predicted_rating"][df["predicted_rating"] > 0]
        if not valid_ratings.empty:
            print(f"Mean Predicted Rating : {valid_ratings.mean():.2f}")
            print(f"Min / Max Rating      : {valid_ratings.min():.2f} / {valid_ratings.max():.2f}")
    if "prediction_latency_ms" in df.columns:
        print(f"Mean Latency (ms)     : {df['prediction_latency_ms'].mean():.2f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
