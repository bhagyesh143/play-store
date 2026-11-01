
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_model():

    data_path = (
        "data/processed/"
        "processed_googleplaystore.csv"
    )

    model_path = (
        "models/"
        "best_google_play_rating_model.pkl"
    )

    df = pd.read_csv(data_path)

    X = df.drop(
        columns=["Rating"]
    )

    y = df["Rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = joblib.load(
        model_path
    )

    predictions = model.predict(
        X_test
    )

    # ======================================================
    # REGRESSION METRICS
    # ======================================================

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    # ======================================================
    # CUSTOM REGRESSION ACCURACY
    # ======================================================

    mean_actual_rating = y_test.mean()

    regression_accuracy = (
        1 -
        (mae / mean_actual_rating)
    ) * 100

    # Keep accuracy between 0 and 100

    regression_accuracy = max(
        0,
        min(
            100,
            regression_accuracy
        )
    )

    # ======================================================
    # DISPLAY RESULTS
    # ======================================================

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(
        "MAE                 :",
        round(mae, 4)
    )

    print(
        "RMSE                :",
        round(rmse, 4)
    )

    print(
        "R2                  :",
        round(r2, 4)
    )

    print(
        "Regression Accuracy :",
        round(regression_accuracy, 2),
        "%"
    )

    print("=" * 60)

    # ======================================================
    # CREATE OUTPUT DIRECTORY
    # ======================================================

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    # ======================================================
    # SAVE PREDICTIONS
    # ======================================================

    results = pd.DataFrame({

        "Actual": y_test.values,

        "Predicted": predictions

    })

    results.to_csv(
        "outputs/predictions.csv",
        index=False
    )

    # ======================================================
    # SAVE METRICS
    # ======================================================

    metrics = pd.DataFrame({

        "Metric": [
            "MAE",
            "RMSE",
            "R2",
            "Regression Accuracy"
        ],

        "Value": [
            mae,
            rmse,
            r2,
            regression_accuracy
        ]

    })

    metrics.to_csv(
        "outputs/model_metrics.csv",
        index=False
    )

    print(
        "\nResults saved to:"
    )

    print(
        "outputs/predictions.csv"
    )

    print(
        "outputs/model_metrics.csv"
    )


if __name__ == "__main__":

    evaluate_model()
