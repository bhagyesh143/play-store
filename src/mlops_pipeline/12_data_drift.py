import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp

BASELINE_PATH = Path("data/processed/X_test_raw.csv")
EVOLVED_PATH = Path("data/monitoring/drift/evolved_data.csv")
REPORT_PATH = Path("outputs/drift/data_drift_report.csv")

NUMERICAL_COLUMNS = ["Reviews", "Size", "Installs", "Price"]
CATEGORICAL_COLUMNS = ["Category", "Type", "Content Rating"]

KS_THRESHOLD = 0.10
JS_THRESHOLD = 0.10


def calculate_js_divergence(baseline, evolved):
    categories = sorted(set(baseline.dropna().unique()) | set(evolved.dropna().unique()))
    p = baseline.value_counts(normalize=True).reindex(categories, fill_value=0).values.astype(float)
    q = evolved.value_counts(normalize=True).reindex(categories, fill_value=0).values.astype(float)

    p /= (p.sum() + 1e-9)
    q /= (q.sum() + 1e-9)
    m = 0.5 * (p + q)

    p_mask = p > 0
    q_mask = q > 0

    kl_pm = np.sum(p[p_mask] * np.log2(p[p_mask] / m[p_mask]))
    kl_qm = np.sum(q[q_mask] * np.log2(q[q_mask] / m[q_mask]))
    return float(0.5 * (kl_pm + kl_qm))


def analyze_drift():
    if not BASELINE_PATH.exists() or not EVOLVED_PATH.exists():
        raise FileNotFoundError("Baseline or evolved dataset not found.")

    baseline = pd.read_csv(BASELINE_PATH)
    evolved = pd.read_csv(EVOLVED_PATH)

    results = []
    # Numerical features
    for col in NUMERICAL_COLUMNS:
        if col in baseline.columns and col in evolved.columns:
            stat, p_val = ks_2samp(baseline[col].dropna(), evolved[col].dropna())
            status = "DRIFT" if stat >= KS_THRESHOLD else "OK"
            results.append({
                "feature": col,
                "type": "numerical",
                "metric": "KS",
                "score": round(float(stat), 6),
                "p_value": round(float(p_val), 6),
                "threshold": KS_THRESHOLD,
                "status": status,
            })

    # Categorical features
    for col in CATEGORICAL_COLUMNS:
        if col in baseline.columns and col in evolved.columns:
            score = calculate_js_divergence(baseline[col], evolved[col])
            status = "DRIFT" if score >= JS_THRESHOLD else "OK"
            results.append({
                "feature": col,
                "type": "categorical",
                "metric": "JS",
                "score": round(float(score), 6),
                "p_value": None,
                "threshold": JS_THRESHOLD,
                "status": status,
            })

    report = pd.DataFrame(results)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)

    print("=" * 70)
    print("DATA DRIFT ANALYSIS (STATISTICAL KS & JS TESTS)")
    print("=" * 70)
    print(report.to_string(index=False))
    drift_count = (report["status"] == "DRIFT").sum()
    print("=" * 70)
    print(f"Features analyzed : {len(report)}")
    print(f"Drift detected    : {drift_count}")
    print(f"Report saved      : {REPORT_PATH}")
    return report


if __name__ == "__main__":
    analyze_drift()
