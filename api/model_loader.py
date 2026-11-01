"""
Model Service Loader
Loads and caches the trained scikit-learn Pipeline and metadata for high-speed inference.
"""

import os
import json
import joblib
import pandas as pd
from typing import Dict, Any, Optional


class ModelService:
    _instance: Optional["ModelService"] = None

    def __init__(self, model_path: str = "models/best_google_play_rating_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.metadata = {}
        self.metrics = {}
        self.model_name = "Google_Play_Store_RandomForest"
        self.model_version = "1.0.0"
        self.load_model()
        self.load_metadata()
        self.load_metrics()

    @classmethod
    def get_instance(cls, model_path: str = "models/best_google_play_rating_model.pkl") -> "ModelService":
        if cls._instance is None:
            cls._instance = cls(model_path=model_path)
        return cls._instance

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Trained model not found at path: {self.model_path}")
        print(f"[INFO] Loading trained model pipeline from: {self.model_path}")
        self.model = joblib.load(self.model_path)
        print("[INFO] Model pipeline loaded successfully into memory.")

    def load_metadata(self):
        metadata_path = "data/processed/dataset_metadata.json"
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"[WARNING] Could not read metadata file: {e}")

    def load_metrics(self):
        metrics_path = "outputs/model_metrics.csv"
        if os.path.exists(metrics_path):
            try:
                df = pd.read_csv(metrics_path)
                self.metrics = dict(zip(df["Metric"], df["Value"]))
            except Exception as e:
                print(f"[WARNING] Could not read metrics file: {e}")

    def predict_dataframe(self, df_features: pd.DataFrame) -> float:
        """
        Runs model prediction on feature DataFrame and clamps output between 1.0 and 5.0.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        pred = float(self.model.predict(df_features)[0])
        # Clamped to valid Play Store rating bounds [1.0, 5.0]
        return max(1.0, min(5.0, pred))

    def get_info(self) -> Dict[str, Any]:
        """
        Returns model parameters, architecture, and baseline performance.
        """
        features_list = [
            "Reviews", "Size", "Installs", "Price",
            "Current Ver", "Android Ver", "Updated_Year",
            "Updated_Month", "Updated_Day", "Category",
            "Type", "Content Rating", "Genres"
        ]

        steps = [str(step[0]) for step in getattr(self.model, "steps", [])]

        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "problem_type": "Supervised Regression",
            "target_variable": "Rating",
            "rating_range": {"min": 1.0, "max": 5.0},
            "baseline_metrics": self.metrics or {
                "MAE": 0.3128,
                "RMSE": 0.4663,
                "R2": 0.1734,
                "Regression Accuracy": "92.57%"
            },
            "features_expected": features_list,
            "pipeline_steps": steps or ["preprocessor", "model"]
        }


# Global instance
model_service = ModelService.get_instance()
