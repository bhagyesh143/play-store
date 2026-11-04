import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from pathlib import Path

DATA_DRIFT_PATH = Path("outputs/drift/data_drift_report.csv")
PREDICTION_DRIFT_PATH = Path("outputs/drift/prediction_drift_report.csv")
CONCEPT_DRIFT_PATH = Path("outputs/drift/concept_drift_report.csv")
OUTPUT_PATH = Path("outputs/drift/drift_analysis_report.csv")


def load_reports():
    data_drift = pd.read_csv(DATA_DRIFT_PATH)
    prediction_drift = pd.read_csv(PREDICTION_DRIFT_PATH)
    concept_drift = pd.read_csv(CONCEPT_DRIFT_PATH)
    return data_drift, prediction_drift, concept_drift


def main():
    data_drift, prediction_drift, concept_drift = load_reports()

    drifted_features = data_drift[data_drift["status"] == "DRIFT"]
    data_summary = {
        "features_analyzed": len(data_drift),
        "features_with_drift": len(drifted_features),
        "drifted_features": ", ".join(drifted_features["feature"].tolist()),
    }

    pred_row = prediction_drift.iloc[0]
    pred_summary = {
        "prediction_distribution_status": pred_row["prediction_distribution_status"],
        "prediction_distribution_ks": pred_row["prediction_distribution_ks"],
        "probability_distribution_status": pred_row["probability_distribution_status"],
        "probability_distribution_ks": pred_row["probability_distribution_ks"],
    }

    degraded_metrics = concept_drift[concept_drift["status"] == "DRIFT"]
    concept_summary = {
        "metrics_analyzed": len(concept_drift),
        "metrics_degraded": len(degraded_metrics),
        "degraded_metrics": ", ".join(degraded_metrics["metric"].tolist()),
    }

    summary_df = pd.DataFrame([
        {
            "category": "Data Drift",
            "metric": "Features with drift",
            "value": data_summary["features_with_drift"],
            "status": "DRIFT" if data_summary["features_with_drift"] > 0 else "OK",
            "details": data_summary["drifted_features"],
        },
        {
            "category": "Prediction Drift",
            "metric": "Prediction distribution",
            "value": pred_summary["prediction_distribution_ks"],
            "status": pred_summary["prediction_distribution_status"],
            "details": "KS statistic",
        },
        {
            "category": "Prediction Drift",
            "metric": "Probability distribution",
            "value": pred_summary["probability_distribution_ks"],
            "status": pred_summary["probability_distribution_status"],
            "details": "KS statistic",
        },
        {
            "category": "Concept Drift",
            "metric": "Degraded performance metrics",
            "value": concept_summary["metrics_degraded"],
            "status": "DRIFT" if concept_summary["metrics_degraded"] > 0 else "OK",
            "details": concept_summary["degraded_metrics"],
        },
    ])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_PATH, index=False)

    print("=" * 70)
    print("COMBINED DRIFT ANALYSIS SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)
    print(f"[SUCCESS] Combined report saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
