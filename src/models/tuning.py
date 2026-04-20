"""Hyperparameter tuning for LightGBM with Optuna (30 trials, 5-fold CV)."""

import json
import time
from pathlib import Path

import joblib
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURES_PATH = Path("data/features.csv")
MODELS_DIR = Path("models")
BEST_PARAMS_PATH = MODELS_DIR / "best_params.json"
TUNED_MODEL_PATH = MODELS_DIR / "tuned_model.pkl"

NON_FEATURE_COLS = ["customer_id", "signup_date", "last_purchase_date", "country", "gender", "churn"]
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
N_TRIALS = 30


def load_features(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load feature matrix and churn target."""
    df = pd.read_csv(path)
    X = df[[c for c in df.columns if c not in NON_FEATURE_COLS]]
    y = df["churn"]
    return X, y


def make_objective(X_train: pd.DataFrame, y_train: pd.Series):
    """Return an Optuna objective that CV-scores LightGBM on the training set."""
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 2.0, 5.0),
            "random_state": 42,
            "verbosity": -1,
        }
        model = LGBMClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=CV, scoring="roc_auc", n_jobs=-1)
        cv_mean = scores.mean()

        print(
            f"  Trial {trial.number:>2} | "
            f"n_est={params['n_estimators']:>3}  depth={params['max_depth']}  "
            f"lr={params['learning_rate']:.4f}  leaves={params['num_leaves']:>3}  "
            f"spw={params['scale_pos_weight']:.2f}  ->  CV ROC-AUC={cv_mean:.4f}"
        )
        return cv_mean

    return objective


if __name__ == "__main__":
    print(f"Loading features from {FEATURES_PATH} ...")
    X, y = load_features(FEATURES_PATH)
    print(f"  Shape: {X.shape}  |  Churn rate: {y.mean():.2%}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Running Optuna ({N_TRIALS} trials, 5-fold CV, scoring=roc_auc) ...")
    t0 = time.perf_counter()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(make_objective(X_train, y_train), n_trials=N_TRIALS)

    tuning_time = time.perf_counter() - t0
    best_params = study.best_params
    best_cv = study.best_value

    print(f"\nBest CV ROC-AUC : {best_cv:.4f}  (tuning: {tuning_time:.1f}s)")
    print("Best params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # Persist best hyperparameters
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest params saved to {BEST_PARAMS_PATH}")

    # Final model on full training set
    print("\nTraining final model with best params ...")
    final_model = LGBMClassifier(**best_params, random_state=42, verbosity=-1)
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)[:, 1]

    print("\n--- Tuned Model — Test Set ---")
    print(f"  ROC-AUC : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"  F1      : {f1_score(y_test, y_pred):.4f}")
    print(f"  Recall  : {recall_score(y_test, y_pred):.4f}")

    joblib.dump(final_model, TUNED_MODEL_PATH)
    print(f"\nTuned model saved to {TUNED_MODEL_PATH}")
