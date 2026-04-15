from pathlib import Path

import joblib
import pandas as pd

from app.config import settings
from ml.features import FEATURE_COLUMNS


class ChurnPredictor:
    def __init__(self):
        self._model = None

    def _load(self):
        path = Path(settings.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. Run `python ml/train.py` first."
            )
        self._model = joblib.load(path)

    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model

    def _to_features(self, customer) -> pd.DataFrame:
        data = {col: getattr(customer, col, 0) or 0 for col in FEATURE_COLUMNS}
        return pd.DataFrame([data])

    def predict(self, customer) -> dict:
        features = self._to_features(customer)
        proba = float(self.model.predict_proba(features)[0][1])

        if proba < 0.33:
            risk = "low"
            recommendation = "Customer is engaged. Maintain current retention strategy."
        elif proba < 0.66:
            risk = "medium"
            recommendation = "Send a personalized re-engagement email with a discount offer."
        else:
            risk = "high"
            recommendation = "Urgent: trigger a win-back campaign with a significant incentive."

        return {
            "churn_probability": round(proba, 4),
            "churn_risk": risk,
            "recommendation": recommendation,
        }
