"""
Evaluate the trained XGBoost model on the full dataset.

Usage:
    python ml/evaluate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from ml.features import FEATURE_COLUMNS, TARGET_COLUMN

MODEL_PATH = Path("ml/artifacts/xgboost_churn.joblib")
DATA_PATH = Path("data/synthetic_customers.csv")


def evaluate():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python ml/train.py` first.")

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS].fillna(0)
    y = df[TARGET_COLUMN].astype(int)

    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    print("=" * 60)
    print("RETAILAI CHURN MODEL — EVALUATION REPORT")
    print("=" * 60)
    print(f"Dataset size    : {len(df):,}")
    print(f"Churn rate      : {y.mean():.2%}")
    print(f"AUC-ROC         : {roc_auc_score(y, y_pred_proba):.4f}")
    print(f"Avg Precision   : {average_precision_score(y, y_pred_proba):.4f}")
    print()
    print(classification_report(y, y_pred, target_names=["Retained", "Churned"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))


if __name__ == "__main__":
    evaluate()
