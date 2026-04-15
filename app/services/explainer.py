from pathlib import Path

import joblib
import pandas as pd

from app.config import settings
from app.schemas.prediction import ShapExplanation
from ml.features import FEATURE_COLUMNS


class ShapExplainer:
    def __init__(self):
        self._explainer = None

    def _load(self):
        path = Path(settings.shap_explainer_path)
        if not path.exists():
            raise FileNotFoundError(
                f"SHAP explainer not found at {path}. Run `python ml/train.py` first."
            )
        self._explainer = joblib.load(path)

    @property
    def explainer(self):
        if self._explainer is None:
            self._load()
        return self._explainer

    def _to_features(self, customer) -> pd.DataFrame:
        data = {col: getattr(customer, col, 0) or 0 for col in FEATURE_COLUMNS}
        return pd.DataFrame([data])

    def explain(self, customer, top_n: int = 5) -> list[ShapExplanation]:
        features = self._to_features(customer)
        shap_values = self.explainer(features).values[0]

        explanations = [
            ShapExplanation(
                feature=col,
                value=float(features[col].iloc[0]),
                shap_value=round(float(shap_val), 4),
                impact="positive" if shap_val > 0 else "negative",
            )
            for col, shap_val in zip(FEATURE_COLUMNS, shap_values)
        ]

        explanations.sort(key=lambda x: abs(x.shap_value), reverse=True)
        return explanations[:top_n]
