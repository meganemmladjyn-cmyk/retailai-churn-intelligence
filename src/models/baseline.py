"""Logistic regression baseline for churn prediction."""

import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = Path("data/features.csv")
MODEL_PATH = Path("models/baseline.pkl")

NON_FEATURE_COLS = ["customer_id", "signup_date", "last_purchase_date", "country", "gender", "churn"]


def load_features(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load feature matrix and target from CSV."""
    df = pd.read_csv(path)
    X = df[[c for c in df.columns if c not in NON_FEATURE_COLS]]
    y = df["churn"]
    return X, y


def train(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit logistic regression baseline with standard scaling."""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(X_train, y_train)
    return model


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Print accuracy, precision, recall, F1, and ROC-AUC on test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- Baseline Metrics ---")
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"  F1        : {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_test, y_proba):.4f}")


if __name__ == "__main__":
    start = time.perf_counter()

    print(f"Loading features from {FEATURES_PATH} ...")
    X, y = load_features(FEATURES_PATH)
    print(f"  Dataset shape: {X.shape}  |  Churn rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

    print("\nTraining LogisticRegression ...")
    model = train(X_train, y_train)

    evaluate(model, X_test, y_test)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    print(f"Done in {time.perf_counter() - start:.2f}s")
