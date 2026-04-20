"""Model comparison: Logistic Regression vs XGBoost vs LightGBM."""

import time
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FEATURES_PATH = Path("data/features.csv")
MODELS_DIR = Path("models")

NON_FEATURE_COLS = ["customer_id", "signup_date", "last_purchase_date", "country", "gender", "churn"]

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def load_features(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load feature matrix and churn target."""
    df = pd.read_csv(path)
    X = df[[c for c in df.columns if c not in NON_FEATURE_COLS]]
    y = df["churn"]
    return X, y


def build_candidates() -> dict[str, object]:
    """Return the three candidate estimators keyed by display name."""
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=3,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            scale_pos_weight=3,
            random_state=42,
            verbosity=-1,
        ),
    }


def evaluate_model(
    name: str,
    model: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Run CV + final fit, return metrics dict."""
    # 5-fold CV on training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=CV, scoring="roc_auc", n_jobs=-1)

    # Final fit + test evaluation
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "Model": name,
        "CV ROC-AUC (mean)": round(cv_scores.mean(), 4),
        "CV ROC-AUC (std)": round(cv_scores.std(), 4),
        "Test ROC-AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Test F1": round(f1_score(y_test, y_pred), 4),
        "Test Recall": round(recall_score(y_test, y_pred), 4),
        "Train time (s)": round(train_time, 2),
        "_model": model,
    }


if __name__ == "__main__":
    print(f"Loading features from {FEATURES_PATH} ...")
    X, y = load_features(FEATURES_PATH)
    print(f"  Shape: {X.shape}  |  Churn rate: {y.mean():.2%}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = []
    for name, model in build_candidates().items():
        print(f"Evaluating {name} ...")
        row = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        results.append(row)
        print(f"  CV ROC-AUC: {row['CV ROC-AUC (mean)']:.4f} ± {row['CV ROC-AUC (std)']:.4f}"
              f"  |  Test ROC-AUC: {row['Test ROC-AUC']:.4f}"
              f"  |  F1: {row['Test F1']:.4f}"
              f"  |  Recall: {row['Test Recall']:.4f}"
              f"  |  {row['Train time (s)']}s")

    # Comparison table
    display_cols = ["Model", "CV ROC-AUC (mean)", "CV ROC-AUC (std)",
                    "Test ROC-AUC", "Test F1", "Test Recall", "Train time (s)"]
    df_results = pd.DataFrame(results)[display_cols].sort_values("CV ROC-AUC (mean)", ascending=False)

    print("\n" + "=" * 78)
    print("  MODEL COMPARISON")
    print("=" * 78)
    print(df_results.to_string(index=False))
    print("=" * 78)

    # Best model
    best = df_results.iloc[0]
    print(f"\nBest model: {best['Model']}")
    print(f"  CV ROC-AUC : {best['CV ROC-AUC (mean)']} ± {best['CV ROC-AUC (std)']}")
    print(f"  Test ROC-AUC: {best['Test ROC-AUC']}  |  F1: {best['Test F1']}  |  Recall: {best['Test Recall']}")

    # Save all models
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    name_to_file = {
        "LogisticRegression": "baseline.pkl",
        "XGBoost": "xgb_churn.pkl",
        "LightGBM": "lgbm_churn.pkl",
    }
    for row in results:
        out = MODELS_DIR / name_to_file[row["Model"]]
        joblib.dump(row["_model"], out)
        print(f"Saved {row['Model']} -> {out}")
