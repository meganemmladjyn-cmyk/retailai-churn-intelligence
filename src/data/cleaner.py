"""
Data cleaning pipeline for the RetailAI Churn Intelligence System.

Applies a sequence of deterministic cleaning steps to the raw customer
DataFrame and validates the result with the data-quality gate.

Usage:
    python src/data/cleaner.py

Output:
    data/processed/customers_clean.csv
"""

import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

# Allow running as a standalone script from any working directory
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.quality import check_data_quality  # noqa: E402

_RAW_CSV = Path("data/raw/customers.csv")
_PROCESSED_CSV = Path("data/processed/customers_clean.csv")

# ---------------------------------------------------------------------------
# Internal step helpers  (each returns the mutated df + rows removed)
# ---------------------------------------------------------------------------


def _drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop duplicate rows, keeping the first occurrence.

    Args:
        df: DataFrame to deduplicate.

    Returns:
        Tuple of (deduplicated DataFrame, number of rows removed).
    """
    before = len(df)
    df = df.drop_duplicates(keep="first")
    return df, before - len(df)


def _drop_null_churn(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove rows where the churn label is null.

    Args:
        df: DataFrame to filter.

    Returns:
        Tuple of (filtered DataFrame, number of rows removed).
    """
    before = len(df)
    df = df.dropna(subset=["churn"])
    return df, before - len(df)


def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values for age, gender, and country.

    - ``age``    : filled with the column median (float).
    - ``gender`` : filled with the column mode (most frequent value).
    - ``country``: filled with the column mode (most frequent value).

    Imputation statistics are computed from the rows present at the time
    this function is called (i.e. after deduplication and churn-null removal).

    Args:
        df: DataFrame with potential missing values in age / gender / country.

    Returns:
        DataFrame with the three columns fully populated.
    """
    df = df.copy()
    df["age"] = df["age"].fillna(df["age"].median())
    df["gender"] = df["gender"].fillna(df["gender"].mode().iloc[0])
    df["country"] = df["country"].fillna(df["country"].mode().iloc[0])
    return df


def _convert_support_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """Convert support_tickets from float64 to int.

    Any residual NaN values are treated as 0 before casting.

    Args:
        df: DataFrame whose support_tickets column may be float64.

    Returns:
        DataFrame with support_tickets as int64.
    """
    df = df.copy()
    df["support_tickets"] = df["support_tickets"].fillna(0).round().astype(int)
    return df


def _remove_negative_spend(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove rows with a negative total_spent value.

    Args:
        df: DataFrame to filter.

    Returns:
        Tuple of (filtered DataFrame, number of rows removed).
    """
    before = len(df)
    df = df[df["total_spent"] >= 0.0]
    return df, before - len(df)


def _clip_return_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Clip return_rate to the valid range [0.0, 1.0].

    Values below 0 are raised to 0; values above 1 are lowered to 1.
    No rows are removed.

    Args:
        df: DataFrame whose return_rate column may contain out-of-range values.

    Returns:
        DataFrame with return_rate clipped to [0.0, 1.0].
    """
    df = df.copy()
    df["return_rate"] = df["return_rate"].clip(lower=0.0, upper=1.0)
    return df


def _remove_future_signups(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove rows whose signup_date is strictly in the future.

    The column is parsed via pd.to_datetime; rows with unparseable dates
    are treated as valid (not removed) to avoid silent data loss.

    Args:
        df: DataFrame to filter.

    Returns:
        Tuple of (filtered DataFrame, number of rows removed).
    """
    before = len(df)
    today = pd.Timestamp(date.today())
    parsed = pd.to_datetime(df["signup_date"], errors="coerce")
    # Keep rows where the date is valid AND not in the future,
    # or where the date could not be parsed (coerced to NaT).
    mask = parsed.isna() | (parsed <= today)
    df = df[mask]
    return df, before - len(df)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the full cleaning pipeline to a raw customer DataFrame.

    Steps applied in order:

    1. Drop duplicate rows (keep first occurrence).
    2. Drop rows where the ``churn`` label is null.
    3. Impute missing ``age`` with the column median; ``gender`` and
       ``country`` with their respective column modes.
    4. Convert ``support_tickets`` from float64 to int.
    5. Remove rows with negative ``total_spent``.
    6. Clip ``return_rate`` to [0.0, 1.0].
    7. Remove rows with ``signup_date`` in the future.
    8. Save the cleaned DataFrame to ``data/processed/customers_clean.csv``.
    9. Run :func:`~src.data.quality.check_data_quality` on the cleaned data.

    Args:
        df: Raw customer DataFrame (typically loaded from
            ``data/raw/customers.csv``).

    Returns:
        Tuple ``(cleaned_df, quality_result)`` where *quality_result* is the
        dict returned by :func:`~src.data.quality.check_data_quality`.
    """
    steps: list[tuple[str, int]] = []

    df, n = _drop_duplicates(df)
    steps.append(("Duplicate rows removed", n))

    df, n = _drop_null_churn(df)
    steps.append(("Null-churn rows removed", n))

    df = _impute_missing(df)
    steps.append(("Missing values imputed (age/gender/country)", 0))

    df = _convert_support_tickets(df)
    steps.append(("support_tickets cast to int", 0))

    df, n = _remove_negative_spend(df)
    steps.append(("Negative total_spent rows removed", n))

    df = _clip_return_rate(df)
    steps.append(("return_rate clipped to [0, 1]", 0))

    df, n = _remove_future_signups(df)
    steps.append(("Future signup_date rows removed", n))

    # Persist cleaned dataset
    _PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = df.reset_index(drop=True)
    df.to_csv(_PROCESSED_CSV, index=False)

    # Validate the cleaned dataset
    quality_result = check_data_quality(df)
    quality_result["cleaning_steps"] = steps  # attach for downstream reporting

    return df, quality_result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _RAW_CSV.exists():
        print(f"ERROR: '{_RAW_CSV}' not found. Run 'python data/generate.py' first.")
        sys.exit(1)

    df_raw = pd.read_csv(_RAW_CSV)
    n_raw = len(df_raw)

    sep = "=" * 54
    print(sep)
    print("  RetailAI - Data cleaning pipeline")
    print(sep)
    print(f"  Source : {_RAW_CSV}")
    print(f"  Output : {_PROCESSED_CSV}")
    print(f"\nRows before cleaning : {n_raw:,}")

    df_clean, quality = clean_data(df_raw)
    n_clean = len(df_clean)

    print(f"Rows after cleaning  : {n_clean:,}  (removed {n_raw - n_clean:,})\n")

    # Step-by-step breakdown
    steps = quality.pop("cleaning_steps", [])
    print("Cleaning steps")
    print("-" * 54)
    for label, dropped in steps:
        suffix = f"  (-{dropped:,})" if dropped else ""
        print(f"  {label:<42}{suffix}")

    # Quality gate result
    status = "PASS" if quality["success"] else "FAIL"
    print(f"\nQuality gate: {status}")

    failures = quality["failures"]
    warnings_list = quality["warnings"]

    if not failures and not warnings_list:
        print("  All checks passed with no issues.")
    else:
        if failures:
            print(f"\nCritical failures ({len(failures)})")
            print("-" * 54)
            for msg in failures:
                print(f"  [FAIL] {msg}")
        if warnings_list:
            print(f"\nWarnings ({len(warnings_list)})")
            print("-" * 54)
            for msg in warnings_list:
                print(f"  [WARN] {msg}")

    print()
    sys.exit(0 if quality["success"] else 1)
