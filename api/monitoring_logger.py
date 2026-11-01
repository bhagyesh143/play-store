"""
Inference Logger for MLOps Prediction Monitoring
Records real-time predictions to structured JSON Lines logs (prediction_logs.jsonl)
"""

import os
import json
import threading
from datetime import datetime
from typing import Dict, Any


class InferenceLogger:
    """
    Thread-safe JSON Lines logger for tracking production inference requests.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, log_path: str = "logs/inference/prediction_logs.jsonl"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(InferenceLogger, cls).__new__(cls)
                cls._instance.log_path = log_path
                cls._instance._init_logger()
            return cls._instance

    def _init_logger(self):
        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def log_prediction(
        self,
        request_id: str,
        input_features: Dict[str, Any],
        predicted_rating: float,
        prediction_latency_ms: float,
        model_name: str = "RandomForestRegressor",
        model_version: str = "1.0.0",
        status: str = "SUCCESS",
        client_ip: str = "unknown"
    ) -> None:
        """
        Appends an inference record to the JSON Lines log file.
        """
        record = {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_name": model_name,
            "model_version": model_version,
            "input_features": input_features,
            "predicted_rating": round(float(predicted_rating), 3),
            "prediction_latency_ms": round(float(prediction_latency_ms), 2),
            "status": status,
            "client_ip": client_ip
        }

        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")


# Global singleton instance
inference_logger = InferenceLogger()
