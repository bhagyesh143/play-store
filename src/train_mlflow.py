
import os
import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_PATH = "/Users/bhagyeshannamareddy/Desktop/play-store/data/processed/processed_googleplaystore.csv"

TEST_SIZE = 0.20
RANDOM_STATE = 42

MLFLOW_EXPERIMENT = "Google_Play_Store_ML_Experiment"


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("[INFO] Loading Google Play Store dataset...")

df = pd.read_csv(DATA_PATH)

print("[INFO] Dataset shape:", df.shape)


# ============================================================
# 3. DATA CLEANING
# ============================================================

print("[INFO] Cleaning dataset...")

# Remove duplicate rows
df = df.drop_duplicates()

# Convert Rating to numeric
df["Rating"] = pd.to_numeric(
    df["Rating"],
    errors="coerce"
)

# Remove rows where Rating is missing
df = df.dropna(
    subset=["Rating"]
)

# Create target
#
# 1 = Rating < 4.0
# 0 = Rating >= 4.0
#
# NOTE:
# This is a proxy target for the MLOps lab.
# It is NOT real customer churn.

df["churn"] = (
    df["Rating"] < 4.0
).astype(int)


print("[INFO] Target distribution:")
print(df["churn"].value_counts())


# ============================================================
# 4. FEATURE SELECTION
# ============================================================

# Rating is used to create the target.
# Therefore Rating must NOT be used as an input feature.

X = df.drop(
    columns=[
        "Rating",
        "churn"
    ]
)

y = df["churn"]


# Remove App because it is mainly an application name
# rather than a useful predictive feature.

if "App" in X.columns:
    X = X.drop(
        columns=["App"]
    )


# ============================================================
# 5. CONVERT NUMERIC COLUMNS
# ============================================================

# Reviews

if "Reviews" in X.columns:

    X["Reviews"] = pd.to_numeric(
        X["Reviews"],
        errors="coerce"
    )


# Size

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


# Installs

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


# Price

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
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(
    "[INFO] Training samples:",
    len(X_train)
)

print(
    "[INFO] Testing samples:",
    len(X_test)
)


# ============================================================
# 7. IDENTIFY FEATURE TYPES
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


print(
    "[INFO] Numeric features:",
    numeric_features
)

print(
    "[INFO] Categorical features:",
    categorical_features
)


# ============================================================
# 8. NUMERICAL PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
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
# 9. CATEGORICAL PREPROCESSING
# ============================================================

categorical_transformer = Pipeline(
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
# 10. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),

        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# 11. DEFINE MODELS
# ============================================================

models = {

    "Logistic_Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),

    "Decision_Tree":
        DecisionTreeClassifier(
            max_depth=10,
            random_state=RANDOM_STATE
        ),

    "Random_Forest":
        RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_STATE
        )
}


# ============================================================
# 12. CREATE / SELECT MLFLOW EXPERIMENT
# ============================================================

mlflow.set_experiment(
    MLFLOW_EXPERIMENT
)


# ============================================================
# 13. TRAIN AND TRACK MODELS
# ============================================================

for model_name, model in models.items():

    print(
        "\n[INFO] ========================================="
    )

    print(
        f"[INFO] Training {model_name}..."
    )

    print(
        "[INFO] ========================================="
    )


    # --------------------------------------------------------
    # Create pipeline
    # --------------------------------------------------------

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )
        ]
    )


    # --------------------------------------------------------
    # Start MLflow run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name=model_name
    ):

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        pipeline.fit(
            X_train,
            y_train
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        y_pred = pipeline.predict(
            X_test
        )


        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )


        # ----------------------------------------------------
        # Print metrics
        # ----------------------------------------------------

        print(
            f"[INFO] Accuracy  : {accuracy:.4f}"
        )

        print(
            f"[INFO] Precision : {precision:.4f}"
        )

        print(
            f"[INFO] Recall    : {recall:.4f}"
        )

        print(
            f"[INFO] F1 Score  : {f1:.4f}"
        )


        # ----------------------------------------------------
        # Log parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "model",
            model_name
        )

        mlflow.log_param(
            "test_size",
            TEST_SIZE
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE
        )

        mlflow.log_param(
            "dataset",
            "Google Play Store"
        )

        mlflow.log_param(
            "dataset_path",
            DATA_PATH
        )

        mlflow.log_param(
            "target",
            "churn"
        )

        mlflow.log_param(
            "target_definition",
            "1 if Rating < 4.0, otherwise 0"
        )


        # ----------------------------------------------------
        # Log metrics
        # ----------------------------------------------------

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )


        # ----------------------------------------------------
        # Log model
        # ----------------------------------------------------
        #
        # MLflow/skops requires explicit trust for these
        # locally-created model types.
        #

        mlflow.sklearn.log_model(
            pipeline,
            name="model",

            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree"
            ]
        )


        print(
            f"[SUCCESS] {model_name} tracked in MLflow."
        )


# ============================================================
# 14. COMPLETE
# ============================================================

print(
    "\n[INFO] ========================================="
)

print(
    "[SUCCESS] All models trained and tracked!"
)

print(
    "[INFO] MLflow Experiment:"
)

print(
    f"[INFO] {MLFLOW_EXPERIMENT}"
)

print(
    "[INFO] ========================================="
)

