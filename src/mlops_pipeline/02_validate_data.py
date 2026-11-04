import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os
import sys
import pandas as pd
import yaml
import pandera.pandas as pa
from pandera import Column, Check


def get_playstore_schema():
    return pa.DataFrameSchema(
        {
            "App": Column(pa.String, nullable=False),
            "Category": Column(pa.String, nullable=False),
            "Rating": Column(pa.Float, nullable=True),
            "Reviews": Column(pa.String, nullable=False),
            "Size": Column(pa.String, nullable=False),
            "Installs": Column(pa.String, nullable=False),
            "Type": Column(pa.String, nullable=True),
            "Price": Column(pa.String, nullable=False),
            "Content Rating": Column(pa.String, nullable=True),
            "Genres": Column(pa.String, nullable=False),
            "Last Updated": Column(pa.String, nullable=False),
            "Current Ver": Column(pa.String, nullable=True),
            "Android Ver": Column(pa.String, nullable=True),
        },
        strict=False,
    )


def validate_schema(df, output_report_name="baseline_validation.csv"):
    print(f"[INFO] Validating Google Play Store schema (Records: {len(df)})...")
    schema = get_playstore_schema()

    # Pre-check: rating bounds for non-null values
    ratings_numeric = pd.to_numeric(df["Rating"], errors="coerce")
    invalid_ratings = ratings_numeric[(ratings_numeric < 1.0) | (ratings_numeric > 5.0)].dropna()

    os.makedirs("artifacts", exist_ok=True)
    report_path = os.path.join("artifacts", output_report_name)

    try:
        schema.validate(df, lazy=True)
        print("[SUCCESS] Schema Validation PASSED. Required columns and types are compliant.")

        # Save clean validation summary
        summary = {
            "validation_status": "PASSED",
            "total_records": len(df),
            "columns_checked": len(schema.columns),
            "invalid_rating_count": len(invalid_ratings),
            "missing_ratings": int(df["Rating"].isnull().sum()),
        }
        pd.DataFrame([summary]).to_csv(report_path, index=False)
        print(f"[INFO] Baseline validation report saved to '{report_path}'.")
        return True
    except pa.errors.SchemaErrors as err:
        print("[ERROR] Schema Validation FAILED. Structural corruptions detected.")
        failures = err.failure_cases[["schema_context", "column", "check", "failure_case", "index"]]
        print(failures.head(10).to_string())

        failures.to_csv(report_path, index=False)
        print(f"[INFO] Detailed failure report saved to '{report_path}'.")
        return False


if __name__ == "__main__":
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = config["data"]["raw_path"]

    if not os.path.exists(data_path):
        print(f"[ERROR] Target file not found at: {data_path}")
        sys.exit(1)

    raw_df = pd.read_csv(data_path)
    passed = validate_schema(raw_df, output_report_name="baseline_validation.csv")
    sys.exit(0 if passed else 1)
