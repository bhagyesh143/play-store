import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import os
import sys
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def run_evaluation():
    print("[INFO] Starting ML Pipeline Model Evaluation...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_path = config["paths"]["model"]
    processed_dir = config["data"]["processed_dir"]

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}. Run 05_train_registry.py first.")
        return False

    try:
        X_test = np.load(os.path.join(processed_dir, "X_test_final.npy"))
        y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    except FileNotFoundError:
        print("[ERROR] Test data not found. Run 03_preprocessing.py first.")
        return False

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    print("\n--- ML Pipeline: Model Evaluation ---")
    for name, value in metrics.items():
        print(f"{name.replace('_', ' ').title():12}: {value:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print("--------------------------------------\n")

    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Plot Confusion Matrix
    cm_path = config["paths"].get("confusion_matrix", "artifacts/confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, f'{z:d}', ha='center', va='center', color='black' if z < cm.max()/2 else 'white')
    plt.title("Confusion Matrix - Play Store App Rating Churn")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    # Plot ROC Curve
    roc_path = config["paths"].get("roc_curve", "artifacts/roc_curve.png")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {metrics['roc_auc']:.3f})", color="darkorange", lw=2)
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()

    report = {
        "stage": "ML Pipeline - Model Evaluation",
        "evaluated_on": "held-out test split",
        "test_rows": int(len(y_test)),
        "metrics": {name: round(value, 4) for name, value in metrics.items()},
        "confusion_matrix": cm.tolist(),
        "artifacts": {
            "confusion_matrix_plot": cm_path,
            "roc_curve_plot": roc_path,
        }
    }

    report_path = config["paths"]["evaluation_report"]
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"[SUCCESS] Evaluation report saved to: {report_path}")
    print(f"[SUCCESS] Confusion matrix plot saved to: {cm_path}")
    print(f"[SUCCESS] ROC curve plot saved to: {roc_path}")
    return True


if __name__ == "__main__":
    passed = run_evaluation()
    sys.exit(0 if passed else 1)
