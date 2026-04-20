"""Train XGBoost churn prediction model and evaluate on held-out test set."""

import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

FEATURES_PATH = Path("data/features.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "xgb_churn.pkl"

NON_FEATURE_COLS = ["customer_id", "signup_date", "last_purchase_date", "country", "gender", "churn"]


def load_features(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load feature matrix and target from CSV.

    Returns:
        Tuple of (X, y) where X contains feature columns and y is the churn label.
    """
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols]
    y = df["churn"]
    return X, y


def train(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """Fit XGBoost classifier with fixed hyperparameters."""
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=3,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Print classification report and ROC-AUC score on test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {auc:.4f}")


if __name__ == "__main__":
    start = time.perf_counter()

    print(f"Loading features from {FEATURES_PATH} ...")
    X, y = load_features(FEATURES_PATH)
    print(f"  Dataset shape: {X.shape}  |  Churn rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

    print("\nTraining XGBoost ...")
    model = train(X_train, y_train)

    evaluate(model, X_test, y_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    elapsed = time.perf_counter() - start
    print(f"Done in {elapsed:.2f}s")
