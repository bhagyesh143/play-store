import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor


def train_model():

    data_path = (
        "data/processed/"
        "processed_googleplaystore.csv"
    )

    df = pd.read_csv(data_path)

    X = df.drop(
        columns=["Rating"]
    )

    y = df["Rating"]

    numerical_features = [
        "Reviews",
        "Size",
        "Installs",
        "Price",
        "Updated_Year",
        "Updated_Month",
        "Updated_Day",
        "Current Ver",
        "Android Ver"
    ]

    categorical_features = [
        "Category",
        "Type",
        "Content Rating",
        "Genres"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
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

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    pipeline.fit(
        X_train,
        y_train
    )

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        pipeline,
        "models/best_google_play_rating_model.pkl"
    )

    print("Model training completed.")
    print(
        "Model saved to:",
        "models/best_google_play_rating_model.pkl"
    )


if __name__ == "__main__":
    train_model()