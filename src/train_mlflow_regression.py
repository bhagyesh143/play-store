"""
Multi-Model Regression Training and MLflow Experiment Tracking Pipeline
Trains Linear Regression, Decision Tree, Random Forest, and Gradient Boosting Regressors.
Logs parameters, metrics (MAE, MSE, RMSE, R2), and artifacts to MLflow.
Selects the best model and registers it in the MLflow Model Registry.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Set MLflow tracking URI to local SQLite database
mlflow.set_tracking_uri("sqlite:///mlflow.db")
EXPERIMENT_NAME = "Google_Play_Store_App_Rating_Regression"
mlflow.set_experiment(EXPERIMENT_NAME)

DATA_PATH = "data/processed/processed_googleplaystore.csv"
MODEL_DIR = "models"
OUTPUTS_DIR = "outputs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def train_and_track_regression_models():
    print("=" * 65)
    print("MLFLOW EXPERIMENT TRACKING: MULTI-MODEL REGRESSION PIPELINE")
    print("=" * 65)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Processed dataset not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded processed dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    X = df.drop(columns=["Rating"])
    y = df["Rating"]

    numerical_features = [
        "Reviews", "Size", "Installs", "Price",
        "Updated_Year", "Updated_Month", "Updated_Day",
        "Current Ver", "Android Ver"
    ]
    categorical_features = [
        "Category", "Type", "Content Rating", "Genres"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="drop"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Candidate regression models
    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(
            max_depth=10, min_samples_split=5, random_state=42
        ),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42
        )
    }

    best_model_name = None
    best_pipeline = None
    best_rmse = float("inf")
    best_metrics = {}
    summary_results = []

    for name, regressor in models.items():
        print(f"\n[INFO] ---> Training & Tracking: {name}...")

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", regressor)
        ])

        with mlflow.start_run(run_name=name) as run:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            mae = float(mean_absolute_error(y_test, predictions))
            mse = float(mean_squared_error(y_test, predictions))
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(y_test, predictions))
            accuracy = max(0.0, min(100.0, (1.0 - (mae / float(y_test.mean()))) * 100.0))

            # Log model parameters
            if hasattr(regressor, "get_params"):
                params = {k: v for k, v in regressor.get_params().items() if isinstance(v, (int, float, str, bool))}
                mlflow.log_params(params)

            # Log metrics
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2_Score", r2)
            mlflow.log_metric("Regression_Accuracy", accuracy)

            # Log pipeline artifact
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                serialization_format="cloudpickle",
                registered_model_name=f"PlayStore_{name}"
            )

            print(f"      MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f} | Accuracy: {accuracy:.2f}%")

            summary_results.append({
                "Model": name,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2": r2,
                "Regression_Accuracy": accuracy,
                "Run_ID": run.info.run_id
            })

            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = name
                best_pipeline = pipeline
                best_metrics = {
                    "MAE": mae,
                    "MSE": mse,
                    "RMSE": rmse,
                    "R2": r2,
                    "Regression Accuracy": f"{accuracy:.2f}%"
                }

    print("\n" + "=" * 65)
    print(f"🏆 BEST MODEL SELECTED: {best_model_name} (Lowest RMSE: {best_rmse:.4f})")
    print("=" * 65)

    # Save best model to models/
    best_model_path = os.path.join(MODEL_DIR, "best_google_play_rating_model.pkl")
    joblib.dump(best_pipeline, best_model_path)
    print(f"[SUCCESS] Saved champion model to: {best_model_path}")

    # Save summary report
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(os.path.join(OUTPUTS_DIR, "model_comparison_report.csv"), index=False)
    print(f"[SUCCESS] Saved model comparison report to: {os.path.join(OUTPUTS_DIR, 'model_comparison_report.csv')}")

    # Update model_metrics.csv
    metrics_records = [{"Metric": k, "Value": v} for k, v in best_metrics.items()]
    pd.DataFrame(metrics_records).to_csv(os.path.join(OUTPUTS_DIR, "model_metrics.csv"), index=False)
    print(f"[SUCCESS] Updated model metrics to: {os.path.join(OUTPUTS_DIR, 'model_metrics.csv')}")

    return summary_df


if __name__ == "__main__":
    train_and_track_regression_models()
