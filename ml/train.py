"""
Train the XGBoost churn model and save artifacts to ml/artifacts/.

Usage:
    python ml/train.py
"""

import sys
from pathlib import Path

import joblib
import shap
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from ml.features import FEATURE_COLUMNS, TARGET_COLUMN

ARTIFACTS_DIR = Path("ml/artifacts")
MODEL_PATH = ARTIFACTS_DIR / "xgboost_churn.joblib"
EXPLAINER_PATH = ARTIFACTS_DIR / "shap_explainer.joblib"
DATA_PATH = Path("data/synthetic_customers.csv")


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run `python data/generate.py` first."
        )
    return pd.read_csv(DATA_PATH)


def train(df: pd.DataFrame) -> tuple:
    X = df[FEATURE_COLUMNS].fillna(0)
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\nAUC-ROC : {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

    explainer = shap.TreeExplainer(model)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(explainer, EXPLAINER_PATH)

    print(f"Model saved      → {MODEL_PATH}")
    print(f"SHAP explainer   → {EXPLAINER_PATH}")

    return model, explainer, auc


if __name__ == "__main__":
    print("Loading dataset...")
    df = load_data()
    print(f"Rows: {len(df):,} | Churn rate: {df[TARGET_COLUMN].mean():.2%}")

    print("\nTraining XGBoost...")
    train(df)
