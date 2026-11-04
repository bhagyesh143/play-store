import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
from pathlib import Path
import pandas as pd

LOG_PATH = Path("logs/inference/prediction_logs.jsonl")


def main():
    print("=" * 60)
    print("INFERENCE REQUEST & AUDIT HISTORY")
    print("=" * 60)

    if not LOG_PATH.exists():
        print(f"[INFO] Log file not found at {LOG_PATH}")
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
        print("[INFO] No records in log file.")
        return

    df = pd.DataFrame(records)
    cols = [c for c in ["request_id", "timestamp", "predicted_rating", "prediction_latency_ms", "status"] if c in df.columns]
    print(df[cols].tail(10).to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
