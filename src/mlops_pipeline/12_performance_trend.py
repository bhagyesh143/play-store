import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
from pathlib import Path
import pandas as pd

REPORT_PATHS = [
    Path("reports/model_evaluation.json"),
    Path("outputs/retraining/retraining_report.json")
]


def main():
    print("=" * 60)
    print("MODEL PERFORMANCE TREND ANALYSIS")
    print("=" * 60)

    records = []
    for p in REPORT_PATHS:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                stage = data.get("stage", p.stem)
                metrics = data.get("metrics", {})
                records.append({"stage": stage, **metrics})

    if records:
        df = pd.DataFrame(records)
        print(df.to_string(index=False))
    else:
        print("[INFO] No previous evaluation checkpoints found.")
    print("=" * 60)


if __name__ == "__main__":
    main()
