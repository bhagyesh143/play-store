
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "data/raw/googleplaystore.csv"

TEST_SIZE = 0.20
RANDOM_STATE = 42

EXPERIMENT_NAME = "Google_Play_Store_Model_Registry"


# ============================================================
# 1. LOAD DATA
# ============================================================

print("[INFO] =========================================")
print("[INFO] Lab 6 - Model Registry")
print("[INFO] =========================================")

print("[INFO] Loading Google Play Store dataset...")

df = pd.read_csv(DATA_PATH)

print("[INFO] Dataset shape:", df.shape)


# ============================================================
# 2. CLEAN DATA
# ============================================================

df = df.drop_duplicates()

df["Rating"] = pd.to_numeric(
    df["Rating"],
    errors="coerce"
)

df = df.dropna(
    subset=["Rating"]
)


# ============================================================
# 3. CREATE TARGET
# ============================================================

# Proxy target:
# 1 = Rating < 4.0
# 0 = Rating >= 4.0

df["churn"] = (
    df["Rating"] < 4.0
).astype(int)


# ============================================================
# 4. FEATURES AND TARGET
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
# 5. CLEAN NUMERIC FEATURES
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
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


# ============================================================
# 7. FEATURE TYPES
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


# ============================================================
# 8. PREPROCESSING
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
# 9. MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=RANDOM_STATE
)


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


# ============================================================
# 10. MLFLOW EXPERIMENT
# ============================================================

mlflow.set_experiment(
    EXPERIMENT_NAME
)


# ============================================================
# 11. TRAIN AND REGISTER MODEL
# ============================================================

print("[INFO] Training Random Forest...")

with mlflow.start_run(
    run_name="Random_Forest_Registry"
) as run:

    pipeline.fit(
        X_train,
        y_train
    )

    y_pred = pipeline.predict(
        X_test
    )

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


    # --------------------------------------------------------
    # Log parameters
    # --------------------------------------------------------

    mlflow.log_param(
        "model",
        "RandomForestClassifier"
    )

    mlflow.log_param(
        "n_estimators",
        100
    )

    mlflow.log_param(
        "max_depth",
        10
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
        "target_definition",
        "1 if Rating < 4.0, otherwise 0"
    )


    # --------------------------------------------------------
    # Log metrics
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Register model
    # --------------------------------------------------------

    model_name = "Google_Play_Store_Random_Forest"

    print(
        "[INFO] Registering model..."
    )

    mlflow.sklearn.log_model(
        pipeline,
        name="model",
        registered_model_name=model_name,
        skops_trusted_types=[
            "numpy.dtype",
            "sklearn.tree._tree.Tree"
        ]
    )


    print(
        "[SUCCESS] Model registered!"
    )

    print(
        "[INFO] Run ID:",
        run.info.run_id
    )

    print(
        "[INFO] Model Name:",
        model_name
    )


print("\n[SUCCESS] Lab 6 model training completed!")
