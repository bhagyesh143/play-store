import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "data/raw/googleplaystore.csv"

def run_lab1():
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 1: Data Preparation & Exploration")
    print("[INFO] Google Play Store Dataset")
    print("[INFO] =========================================")

    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Raw dataset not found at {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded dataset shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")
    print(f"[INFO] Missing values:\n{df.isnull().sum()}")

    # Drop duplicates
    clean_df = df.drop_duplicates()
    print(f"[INFO] Shape after removing duplicates: {clean_df.shape}")

    os.makedirs("artifacts", exist_ok=True)
    summary = {
        "lab": "Lab 1 - Data Preparation",
        "total_records": int(len(df)),
        "unique_records": int(len(clean_df)),
        "columns_count": int(df.shape[1]),
        "columns": list(df.columns)
    }
    pd.DataFrame([summary]).to_csv("artifacts/lab1_data_summary.csv", index=False)
    print("[SUCCESS] Lab 1 data preparation completed!")

if __name__ == "__main__":
    run_lab1()
