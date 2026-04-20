"""MLflow training run: logs baseline, tuned LightGBM, and XGBoost to experiment 'churn-prediction'."""

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Fix Windows cp1252 encoding crash with MLflow emojis — required on Windows machines
sys.stdout.reconfigure(encoding="utf-8")

import joblib
import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = Path("data/features.csv")
MODELS_DIR = Path("models")
BEST_PARAMS_PATH = MODELS_DIR / "best_params.json"
PRODUCTION_MODEL_PATH = MODELS_DIR / "production_model.pkl"

NON_FEATURE_COLS = ["customer_id", "signup_date", "last_purchase_date", "country", "gender", "churn"]

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "churn-prediction"


def load_features(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load feature matrix and churn target."""
    df = pd.read_csv(path)
    X = df[[c for c in df.columns if c not in NON_FEATURE_COLS]]
    y = df["churn"]
    return X, y


def build_model_configs() -> list[dict]:
    """Return the three model configurations to log."""
    best_params = json.loads(BEST_PARAMS_PATH.read_text())

    return [
        {
            "name": "baseline",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ]),
            "params": {"model_type": "LogisticRegression", "max_iter": 1000, "scaler": "StandardScaler"},
        },
        {
            "name": "tuned_best",
            "model": LGBMClassifier(**best_params, random_state=42, verbosity=-1),
            "params": {"model_type": "LightGBM"} | best_params,
        },
        {
            "name": "xgboost",
            "model": joblib.load(MODELS_DIR / "xgb_churn.pkl"),
            "params": {
                "model_type": "XGBoost",
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "scale_pos_weight": 3,
            },
        },
    ]


def train_and_log(
    config: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Fit model, evaluate on test set, log everything to MLflow, return metrics row."""
    name = config["name"]
    model = config["model"]

    with mlflow.start_run(run_name=name):
        # Train
        t0 = time.perf_counter()
        if name != "xgboost":  # XGBoost already trained; re-fit for consistency
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)
        train_time = round(time.perf_counter() - t0, 2)

        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            "test_roc_auc": round(roc_auc_score(y_test, y_proba), 4),
            "test_f1": round(f1_score(y_test, y_pred), 4),
            "test_recall": round(recall_score(y_test, y_pred), 4),
            "train_time_s": train_time,
        }

        # Log to MLflow
        mlflow.log_params(config["params"])
        mlflow.log_metrics(metrics)

        # Save model artifact via joblib into a temp dir so MLflow can archive it
        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / f"{name}.pkl"
            joblib.dump(model, artifact_path)
            mlflow.log_artifact(str(artifact_path), artifact_path="model")

        # Also persist to models/
        joblib.dump(model, MODELS_DIR / f"{name}.pkl")

    return {"Model": name} | metrics


if __name__ == "__main__":
    print(f"Loading features from {FEATURES_PATH} ...")
    X, y = load_features(FEATURES_PATH)
    print(f"  Shape: {X.shape}  |  Churn rate: {y.mean():.2%}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow experiment: '{EXPERIMENT_NAME}'  ({MLFLOW_TRACKING_URI})\n")

    rows = []
    for config in build_model_configs():
        print(f"Running: {config['name']} ...")
        row = train_and_log(config, X_train, y_train, X_test, y_test)
        rows.append(row)
        print(f"  ROC-AUC={row['test_roc_auc']}  F1={row['test_f1']}  Recall={row['test_recall']}  ({row['train_time_s']}s)")

    # Promote tuned_best as production model
    shutil.copy(MODELS_DIR / "tuned_best.pkl", PRODUCTION_MODEL_PATH)
    print(f"\nProduction model -> {PRODUCTION_MODEL_PATH}")

    # Summary table
    df = pd.DataFrame(rows)
    print("\n" + "=" * 66)
    print("  TRAINING SUMMARY")
    print("=" * 66)
    print(df.to_string(index=False))
    print("=" * 66)
    print(f"\nAll runs logged to {MLFLOW_TRACKING_URI}/#/experiments")
