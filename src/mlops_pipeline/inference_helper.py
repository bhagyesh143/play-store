import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

def clean_dataframe_for_inference(df):
    df = df.copy()

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
        df["Size"] = df["Size"].fillna(15.0)

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

    feature_cols = [c for c in df.columns if c not in ["App", "Rating", "churn", "Last Updated", "Current Ver", "Android Ver"]]
    return df[feature_cols]

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

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
