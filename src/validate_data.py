import pandas as pd
import os

DATA_PATH = "data/raw/googleplaystore.csv"

print("[INFO] =========================================")
print("[INFO] Lab 5 - Data Validation")
print("[INFO] =========================================")

print("[INFO] Loading Google Play Store dataset...")

df = pd.read_csv(DATA_PATH)

print("[INFO] Dataset shape:", df.shape)

# Required columns
required_columns = [
    "App",
    "Category",
    "Rating",
    "Reviews",
    "Size",
    "Installs",
    "Type",
    "Price",
    "Content Rating",
    "Genres",
    "Last Updated",
    "Current Ver",
    "Android Ver"
]

print("\n[INFO] Checking required columns...")

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    print("[ERROR] Missing columns:", missing_columns)
    raise SystemExit(1)

print("[SUCCESS] All required columns are present.")


# Check duplicate rows
duplicates = df.duplicated().sum()

print("\n[INFO] Duplicate rows:", duplicates)


# Check Rating
print("\n[INFO] Checking Rating column...")

df["Rating"] = pd.to_numeric(
    df["Rating"],
    errors="coerce"
)

invalid_rating = (
    (df["Rating"] < 0) |
    (df["Rating"] > 5)
).sum()

print("[INFO] Invalid Rating values:", invalid_rating)


# Check missing values
print("\n[INFO] Missing values:")

print(df.isnull().sum())


# Create validation report
os.makedirs("artifacts", exist_ok=True)

report = {
    "dataset": "Google Play Store",
    "rows": int(df.shape[0]),
    "columns": int(df.shape[1]),
    "duplicate_rows": int(duplicates),
    "invalid_rating_values": int(invalid_rating),
    "required_columns_present": len(missing_columns) == 0
}

report_df = pd.DataFrame([report])

report_df.to_csv(
    "artifacts/data_validation_report.csv",
    index=False
)

print("\n[SUCCESS] Data validation completed!")
print("[INFO] Report saved to artifacts/data_validation_report.csv")