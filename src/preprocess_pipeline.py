import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "data/raw/googleplaystore.csv"

MODEL_PATH = "models/preprocessor.pkl"

os.makedirs("models", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)


print("[INFO] =========================================")
print("[INFO] Lab 5 - Reproducible Data Pipeline")
print("[INFO] =========================================")


# ============================================================
# 1. LOAD DATA
# ============================================================

print("[INFO] Loading Google Play Store dataset...")

df = pd.read_csv(DATA_PATH)

print("[INFO] Original shape:", df.shape)


# ============================================================
# 2. REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates()

print("[INFO] Shape after removing duplicates:", df.shape)


# ============================================================
# 3. CONVERT RATING
# ============================================================

df["Rating"] = pd.to_numeric(
    df["Rating"],
    errors="coerce"
)

df = df.dropna(
    subset=["Rating"]
)


# ============================================================
# 4. CREATE TARGET
# ============================================================

# Proxy target:
# 1 = Rating < 4.0
# 0 = Rating >= 4.0

df["churn"] = (
    df["Rating"] < 4.0
).astype(int)


# ============================================================
# 5. CREATE FEATURES
# ============================================================

X = df.drop(
    columns=[
        "Rating",
        "churn"
    ]
)

y = df["churn"]


# Remove App
if "App" in X.columns:
    X = X.drop(
        columns=["App"]
    )


# ============================================================
# 6. CLEAN NUMERIC COLUMNS
# ============================================================

if "Reviews" in X.columns:

    X["Reviews"] = pd.to_numeric(
        X["Reviews"],
        errors="coerce"
    )


if "Size" in X.columns:

    X["Size"] = (
        X["Size"]
        .astype(str)
        .str.replace(
            "M",
            "",
            regex=False
        )
        .str.replace(
            "k",
            "",
            regex=False
        )
    )

    X["Size"] = pd.to_numeric(
        X["Size"],
        errors="coerce"
    )


if "Installs" in X.columns:

    X["Installs"] = (
        X["Installs"]
        .astype(str)
        .str.replace(
            ",",
            "",
            regex=False
        )
        .str.replace(
            "+",
            "",
            regex=False
        )
    )

    X["Installs"] = pd.to_numeric(
        X["Installs"],
        errors="coerce"
    )


if "Price" in X.columns:

    X["Price"] = (
        X["Price"]
        .astype(str)
        .str.replace(
            "$",
            "",
            regex=False
        )
    )

    X["Price"] = pd.to_numeric(
        X["Price"],
        errors="coerce"
    )


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 8. IDENTIFY FEATURE TYPES
# ============================================================

numeric_features = (
    X_train
    .select_dtypes(
        include=["int64", "float64"]
    )
    .columns
    .tolist()
)

categorical_features = (
    X_train
    .select_dtypes(
        include=["object"]
    )
    .columns
    .tolist()
)


print("[INFO] Numeric features:")
print(numeric_features)

print("[INFO] Categorical features:")
print(categorical_features)


# ============================================================
# 9. NUMERIC PIPELINE
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ============================================================
# 10. CATEGORICAL PIPELINE
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# ============================================================
# 11. COLUMN TRANSFORMER
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 12. FIT PREPROCESSOR
# ============================================================

print("[INFO] Fitting preprocessing pipeline...")

preprocessor.fit(X_train)


# ============================================================
# 13. TRANSFORM DATA
# ============================================================

X_train_processed = preprocessor.transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)


# ============================================================
# 14. SAVE PREPROCESSOR
# ============================================================

joblib.dump(
    preprocessor,
    MODEL_PATH
)

print(
    "[SUCCESS] Preprocessor saved to:",
    MODEL_PATH
)


# ============================================================
# 15. SAVE PROCESSED DATA
# ============================================================

np.save(
    "artifacts/X_train.npy",
    X_train_processed.toarray()
    if hasattr(X_train_processed, "toarray")
    else X_train_processed
)

np.save(
    "artifacts/X_test.npy",
    X_test_processed.toarray()
    if hasattr(X_test_processed, "toarray")
    else X_test_processed
)

np.save(
    "artifacts/y_train.npy",
    y_train.values
)

np.save(
    "artifacts/y_test.npy",
    y_test.values
)


# ============================================================
# 16. SAVE PIPELINE SUMMARY
# ============================================================

summary = {
    "dataset": "Google Play Store",
    "original_rows": int(len(df)),
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "random_state": 42,
    "test_size": 0.20,
    "target": "churn",
    "target_definition": "1 if Rating < 4.0, otherwise 0"
}

pd.DataFrame([summary]).to_json(
    "artifacts/preprocessing_summary.json",
    orient="records",
    indent=4
)


print("[SUCCESS] Data preprocessing completed!")
print("[SUCCESS] Reproducible preprocessing pipeline created!")