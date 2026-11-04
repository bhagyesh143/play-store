import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def clean_dataframe(df, config):
    df = df.copy()
    id_column = config["data"].get("id_column", "App")

    # Drop duplicates
    df = df.drop_duplicates()

    # Convert Rating and create target
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df = df.dropna(subset=["Rating"])
    df = df[(df["Rating"] >= 1.0) & (df["Rating"] <= 5.0)]

    target_column = config["data"].get("target_column", "churn")
    df[target_column] = (df["Rating"] < 4.0).astype(int)

    # Clean numeric features
    if "Reviews" in df.columns:
        df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce").fillna(0)

    if "Size" in df.columns:
        def parse_size(val):
            val_str = str(val).strip()
            if val_str.endswith("M"):
                try:
                    return float(val_str[:-1])
                except:
                    return np.nan
            elif val_str.endswith("k"):
                try:
                    return float(val_str[:-1]) / 1024.0
                except:
                    return np.nan
            else:
                return np.nan

        df["Size"] = df["Size"].apply(parse_size)
        median_size = df["Size"].median()
        df["Size"] = df["Size"].fillna(median_size if pd.notnull(median_size) else 15.0)

    if "Installs" in df.columns:
        df["Installs"] = (
            df["Installs"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("+", "", regex=False)
        )
        df["Installs"] = pd.to_numeric(df["Installs"], errors="coerce").fillna(10000)

    if "Price" in df.columns:
        df["Price"] = (
            df["Price"]
            .astype(str)
            .str.replace("$", "", regex=False)
        )
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0.0)

    # Drop App name from features, keep for ID tracking if needed
    feature_cols = [c for c in df.columns if c not in ["App", "Rating", target_column, "Last Updated", "Current Ver", "Android Ver"]]
    return df, feature_cols


def build_preprocessor(X_train):
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ]
    )
    return preprocessor, cat_cols, num_cols


def run_preprocessing():
    print("[INFO] Starting Preprocessing Pipeline...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = config["data"]["raw_path"]
    processed_dir = config["data"]["processed_dir"]
    target_column = config["data"]["target_column"]

    if not os.path.exists(data_path):
        print(f"[ERROR] Raw dataset not found at: {data_path}")
        return False

    raw_df = pd.read_csv(data_path)
    cleaned_df, feature_cols = clean_dataframe(raw_df, config)

    X = cleaned_df[feature_cols]
    y = cleaned_df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        stratify=y,
    )

    preprocessor, cat_cols, num_cols = build_preprocessor(X_train)

    print("[INFO] Fitting preprocessor on training data...")
    X_train_final = preprocessor.fit_transform(X_train)
    X_test_final = preprocessor.transform(X_test)

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    # Save processed numpy arrays
    np.save(os.path.join(processed_dir, "X_train_final.npy"), X_train_final)
    np.save(os.path.join(processed_dir, "X_test_final.npy"), X_test_final)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train.to_numpy(dtype=np.int64))
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test.to_numpy(dtype=np.int64))

    # Also save to artifacts for backward compatibility
    np.save("artifacts/X_train.npy", X_train_final)
    np.save("artifacts/X_test.npy", X_test_final)
    np.save("artifacts/y_train.npy", y_train.to_numpy(dtype=np.int64))
    np.save("artifacts/y_test.npy", y_test.to_numpy(dtype=np.int64))

    # Save raw feature splits
    X_train.to_csv(os.path.join(processed_dir, "X_train_raw.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test_raw.csv"), index=False)

    preprocessor_path = config["paths"]["preprocessor"]
    joblib.dump(preprocessor, preprocessor_path)

    metadata = {
        "dataset_name": "Google Play Store Apps",
        "target_column": target_column,
        "train_shape_raw": list(X_train.shape),
        "test_shape_raw": list(X_test.shape),
        "train_shape_transformed": list(X_train_final.shape),
        "test_shape_transformed": list(X_test_final.shape),
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "split_test_size": config["split"]["test_size"],
        "split_random_state": config["split"]["random_state"],
        "raw_train_path": os.path.join(processed_dir, "X_train_raw.csv"),
        "raw_test_path": os.path.join(processed_dir, "X_test_raw.csv"),
    }

    metadata_path = os.path.join(processed_dir, "dataset_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    summary_report_path = "artifacts/preprocessing_summary_report.json"
    with open(summary_report_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[SUCCESS] Preprocessing complete. Preprocessor saved to {preprocessor_path}")
    print(f"[SUCCESS] Metadata saved to {metadata_path}")
    return True


if __name__ == "__main__":
    passed = run_preprocessing()
    sys.exit(0 if passed else 1)
