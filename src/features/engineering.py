"""Feature engineering pipeline for customer churn prediction."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parents[2] / "data" / "processed" / "customers_clean.csv"


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer churn-predictive features from raw customer data.

    Args:
        df: Raw customer dataframe with columns from customers_clean.csv.

    Returns:
        DataFrame with all original columns plus engineered features.
    """
    out = df.copy()

    now = pd.Timestamp.now()

    # Parse date columns if they come in as strings
    for col in ("signup_date", "last_purchase_date"):
        if out[col].dtype == object:
            out[col] = pd.to_datetime(out[col])

    # ── RFM Features ─────────────────────────────────────────────────────────

    # Recency: binned to avoid leaking the exact churn threshold (>= 90 days)
    _days_since = (now - out["last_purchase_date"]).dt.days
    out["is_recent_buyer"] = (_days_since <= 30).astype(int)
    out["recency_segment"] = pd.cut(
        _days_since,
        bins=[-1, 30, 90, float("inf")],
        labels=[0, 1, 2],  # active / at_risk / churned_risk
    ).astype(int)

    # Customer tenure in months (needed for frequency below)
    out["tenure_months"] = (now - out["signup_date"]).dt.days / 30

    # Frequency: purchase rate per month since signup — lower = more churn risk
    out["purchase_frequency"] = out["total_orders"] / out["tenure_months"].clip(lower=1)

    # Monetary: avg_order_value already exists; add spend per order as sanity alias
    # spend_per_order recalculates from totals (more robust than stored avg)
    out["spend_per_order"] = out["total_spent"] / out["total_orders"].clip(lower=1)

    # ── Engagement Features ───────────────────────────────────────────────────

    # Support pressure: tickets relative to orders — dissatisfied customers churn more
    out["support_ratio"] = out["support_tickets"] / out["total_orders"].clip(lower=1)

    # Return behavior: high return rate signals low satisfaction
    out["high_returner"] = (out["return_rate"] > 0.5).astype(int)

    # ── Risk Score Features ───────────────────────────────────────────────────

    # Low activity flag: few orders correlates with disengagement
    out["low_activity"] = (out["total_orders"] < 5).astype(int)

    # High value customer: top-25% spenders are less likely to churn
    out["high_value"] = (out["total_spent"] > out["total_spent"].quantile(0.75)).astype(int)

    # ── Interaction Features ──────────────────────────────────────────────────

    # RFM combined risk: low frequency + high support = danger zone
    out["rfm_risk_score"] = out["support_ratio"] / out["purchase_frequency"].clip(lower=0.01)

    return out


def select_features(df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """Remove redundant and low-variance features from an engineered dataframe.

    Drops:
    - Features with pairwise Pearson correlation > 0.95 (keeps the first seen).
    - Features whose variance is below 1% of the mean column variance.

    Args:
        df: DataFrame produced by create_features().

    Returns:
        Tuple of (selected_feature_names, reduced_dataframe).
    """
    # Keep only numeric columns (exclude IDs and categoricals)
    exclude_always = {"customer_id", "churn"}

    candidate_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in exclude_always
    ]

    working = df[candidate_cols].copy()

    # ── Variance filter ───────────────────────────────────────────────────────
    # Normalize to [0,1] before computing variance so high-scale columns like
    # total_spent don't inflate the threshold and silently kill binary features.
    col_min = working.min()
    col_range = (working.max() - col_min).replace(0, 1)
    normalized = (working - col_min) / col_range
    variances = normalized.var()
    variance_threshold = 0.01 * variances.mean()
    low_var_cols = variances[variances < variance_threshold].index.tolist()
    if low_var_cols:
        logger.info("Dropped (low variance, threshold=%.4f): %s", variance_threshold, low_var_cols)
    working = working.drop(columns=low_var_cols)

    # ── Correlation filter ────────────────────────────────────────────────────
    corr_matrix = working.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    high_corr_cols = [
        col for col in upper_triangle.columns
        if any(upper_triangle[col] > 0.95)
    ]
    if high_corr_cols:
        for col in high_corr_cols:
            correlated_with = upper_triangle.index[upper_triangle[col] > 0.95].tolist()
            logger.info("Dropped (corr > 0.95 with %s): %s", correlated_with, col)
    working = working.drop(columns=high_corr_cols)

    selected = working.columns.tolist()

    # Reconstruct output with non-feature columns preserved
    passthrough = [c for c in df.columns if c not in candidate_cols]
    reduced_df = pd.concat([df[passthrough], working], axis=1)

    return selected, reduced_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    raw = pd.read_csv(DATA_PATH, parse_dates=["signup_date", "last_purchase_date"])
    print(f"Loaded: {raw.shape[0]:,} rows × {raw.shape[1]} columns")

    engineered = create_features(raw)
    print(f"\nAfter create_features(): {engineered.shape[0]:,} rows × {engineered.shape[1]} columns")

    new_features = [c for c in engineered.columns if c not in raw.columns]
    print(f"New features ({len(new_features)}): {new_features}")

    selected, reduced = select_features(engineered)
    print(f"\nAfter select_features(): {reduced.shape[0]:,} rows × {reduced.shape[1]} columns")
    print(f"Selected features ({len(selected)}): {selected}")

    dropped = [f for f in new_features if f not in selected]
    if dropped:
        print(f"Dropped features: {dropped}")
    else:
        print("No features dropped by select_features().")
