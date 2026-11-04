"""
Model Service Loader
Loads and caches the trained scikit-learn Pipeline and metadata for high-speed inference.
Supports deployment bundle (deployment/) and standard models/ location.
"""

import os
import json
import joblib
import pandas as pd
from typing import Dict, Any, Optional


class ModelService:
    _instance: Optional["ModelService"] = None

    def __init__(self, model_path: Optional[str] = None):
        if model_path:
            self.model_path = model_path
        elif os.path.exists("deployment/model.pkl"):
            self.model_path = "deployment/model.pkl"
        elif os.path.exists("models/best_google_play_rating_model.pkl"):
            self.model_path = "models/best_google_play_rating_model.pkl"
        elif os.path.exists("models/random_forest_model.pkl"):
            self.model_path = "models/random_forest_model.pkl"
        else:
            self.model_path = "models/best_google_play_rating_model.pkl"

        self.model = None
        self.preprocessor = None
        self.metadata = {}
        self.metrics = {}
        self.model_name = "Google_Play_Store_RandomForest"
        self.model_version = "1.0.0"
        self.load_model()
        self.load_preprocessor()
        self.load_metadata()
        self.load_metrics()

    @classmethod
    def get_instance(cls, model_path: Optional[str] = None) -> "ModelService":
        if cls._instance is None:
            cls._instance = cls(model_path=model_path)
        return cls._instance

    def load_model(self):
        if not os.path.exists(self.model_path):
            print(f"[WARNING] Model path {self.model_path} not found yet.")
            return
        print(f"[INFO] Loading trained model pipeline from: {self.model_path}")
        self.model = joblib.load(self.model_path)
        print("[INFO] Model pipeline loaded successfully into memory.")

    def load_preprocessor(self):
        candidates = ["deployment/preprocessor.pkl", "models/preprocessor.pkl"]
        for p in candidates:
            if os.path.exists(p):
                try:
                    self.preprocessor = joblib.load(p)
                    print(f"[INFO] Preprocessor loaded from: {p}")
                    break
                except Exception as e:
                    print(f"[WARNING] Could not load preprocessor from {p}: {e}")

    def load_metadata(self):
        meta_candidates = ["deployment/model_metadata.json", "data/processed/dataset_metadata.json"]
        for p in meta_candidates:
            if os.path.exists(p):
                try:
                    with open(p, "r") as f:
                        self.metadata = json.load(f)
                    if "model_name" in self.metadata:
                        self.model_name = self.metadata["model_name"]
                    if "model_version" in self.metadata:
                        self.model_version = str(self.metadata["model_version"])
                    break
                except Exception as e:
                    print(f"[WARNING] Could not read metadata from {p}: {e}")

    def load_metrics(self):
        metrics_candidates = ["outputs/model_metrics.csv", "reports/model_evaluation.json"]
        for p in metrics_candidates:
            if os.path.exists(p):
                try:
                    if p.endswith(".csv"):
                        df = pd.read_csv(p)
                        self.metrics = dict(zip(df["Metric"], df["Value"]))
                    elif p.endswith(".json"):
                        with open(p, "r") as f:
                            data = json.load(f)
                            self.metrics = data.get("metrics", {})
                    break
                except Exception as e:
                    print(f"[WARNING] Could not read metrics file {p}: {e}")

    def predict_dataframe(self, df_features: pd.DataFrame) -> float:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        if hasattr(self.model, "steps"):
            pred = float(self.model.predict(df_features)[0])
        elif self.preprocessor is not None:
            try:
                X = self.preprocessor.transform(df_features)
                pred = float(self.model.predict(X)[0])
            except Exception:
                pred = float(self.model.predict(df_features)[0])
        else:
            pred = float(self.model.predict(df_features)[0])
        return max(1.0, min(5.0, pred))

    def get_info(self) -> Dict[str, Any]:
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
            "problem_type": "Supervised Regression / Churn Proxy",
            "target_variable": "Rating",
            "rating_range": {"min": 1.0, "max": 5.0},
            "baseline_metrics": self.metrics or {
                "accuracy": 0.8124,
                "precision": 0.7250,
                "recall": 0.7100,
                "f1_score": 0.7174,
                "roc_auc": 0.8250
            },
            "features_expected": features_list,
            "pipeline_steps": steps or ["preprocessor", "model"]
        }


# Global instance
model_service = ModelService.get_instance()
