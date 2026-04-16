"""
Data quality gate for the RetailAI Churn Intelligence System.

Runs five deterministic checks on a customer DataFrame and returns a
structured report that distinguishes critical failures from warnings.

Usage:
    python src/data/quality.py
"""

import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: list[str] = [
    "customer_id",
    "signup_date",
    "last_purchase_date",
    "total_orders",
    "total_spent",
    "avg_order_value",
    "return_rate",
    "support_tickets",
    "country",
    "age",
    "gender",
    "churn",
]

# Columns that must have a numeric dtype when the DataFrame is loaded.
NUMERIC_COLUMNS: list[str] = [
    "total_orders",
    "total_spent",
    "avg_order_value",
    "return_rate",
    "support_tickets",
    "age",
    "churn",
]

_RAW_CSV = Path("data/raw/customers.csv")

# ---------------------------------------------------------------------------
# Individual checks
# Each function mutates the shared failures / warnings / stats containers.
# ---------------------------------------------------------------------------


def _check_schema(
    df: pd.DataFrame,
    failures: list[str],
    warnings: list[str],
    stats: dict[str, Any],
) -> None:
    """Check 1 — required columns are present and numeric columns have numeric dtypes.

    Args:
        df: DataFrame to validate.
        failures: Accumulator for critical error messages.
        warnings: Accumulator for non-critical concern messages.
        stats: Accumulator for counts and metadata.
    """
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    wrong_dtype_cols: list[str] = []

    for col in missing_cols:
        failures.append(f"[Schema] Missing required column: '{col}'.")

    for col in NUMERIC_COLUMNS:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            wrong_dtype_cols.append(col)
            failures.append(
                f"[Schema] '{col}' expected numeric dtype, got '{df[col].dtype}'."
            )

    stats["missing_columns"] = missing_cols
    stats["wrong_dtype_columns"] = wrong_dtype_cols


def _check_row_count(
    df: pd.DataFrame,
    failures: list[str],
    warnings: list[str],
    stats: dict[str, Any],
) -> None:
    """Check 2 — dataset has enough rows for meaningful modelling.

    Critical if < 100 rows. Warning if < 1,000 rows.

    Args:
        df: DataFrame to validate.
        failures: Accumulator for critical error messages.
        warnings: Accumulator for non-critical concern messages.
        stats: Accumulator for counts and metadata.
    """
    n = len(df)
    stats["row_count"] = n

    if n < 100:
        failures.append(
            f"[Row count] {n:,} rows found — minimum required is 100."
        )
    elif n < 1_000:
        warnings.append(
            f"[Row count] {n:,} rows found — recommended minimum is 1,000."
        )


def _check_null_rates(
    df: pd.DataFrame,
    failures: list[str],
    warnings: list[str],
    stats: dict[str, Any],
) -> None:
    """Check 3 — no column is majority-null; columns with high null rates are flagged.

    Critical if a column has > 50 % nulls.
    Warning if a column has > 20 % (and <= 50 %) nulls.

    Args:
        df: DataFrame to validate.
        failures: Accumulator for critical error messages.
        warnings: Accumulator for non-critical concern messages.
        stats: Accumulator for counts and metadata.
    """
    null_rates: dict[str, float] = {}

    for col in df.columns:
        rate = df[col].isna().mean()
        if rate > 0:
            null_rates[col] = round(rate, 4)

        if rate > 0.50:
            failures.append(
                f"[Null rate] '{col}': {rate:.1%} nulls — exceeds 50 % critical threshold."
            )
        elif rate > 0.20:
            warnings.append(
                f"[Null rate] '{col}': {rate:.1%} nulls — exceeds 20 % warning threshold."
            )

    stats["null_rates"] = null_rates


def _check_value_ranges(
    df: pd.DataFrame,
    failures: list[str],
    warnings: list[str],
    stats: dict[str, Any],
) -> None:
    """Check 4 — domain constraints on key numeric and date columns.

    Failures (critical):
        - Any negative ``total_spent`` value.
        - Any ``return_rate`` outside [0, 1].
        - Any ``signup_date`` in the future.

    Warnings:
        - ``support_tickets`` dtype is float instead of int
          (values may still be integer-compatible).

    Args:
        df: DataFrame to validate.
        failures: Accumulator for critical error messages.
        warnings: Accumulator for non-critical concern messages.
        stats: Accumulator for counts and metadata.
    """
    today = date.today()
    issues: dict[str, Any] = {}

    # -- negative total_spent --------------------------------------------------
    if "total_spent" in df.columns:
        n_neg = int((df["total_spent"] < 0).sum())
        issues["negative_total_spent"] = n_neg
        if n_neg > 0:
            failures.append(
                f"[Value range] total_spent: {n_neg:,} negative value(s) found."
            )

    # -- return_rate outside [0, 1] -------------------------------------------
    if "return_rate" in df.columns:
        valid_rr = df["return_rate"].dropna()
        n_invalid = int(((valid_rr < 0) | (valid_rr > 1)).sum())
        issues["invalid_return_rate"] = n_invalid
        if n_invalid > 0:
            failures.append(
                f"[Value range] return_rate: {n_invalid:,} value(s) outside [0, 1]."
            )

    # -- support_tickets must be integer-compatible ---------------------------
    if "support_tickets" in df.columns:
        valid_tickets = df["support_tickets"].dropna()
        n_fractional = int((valid_tickets % 1 != 0).sum())
        is_float_dtype = pd.api.types.is_float_dtype(df["support_tickets"])
        issues["non_integer_support_tickets"] = n_fractional
        issues["support_tickets_float_dtype"] = is_float_dtype

        if n_fractional > 0:
            warnings.append(
                f"[Value range] support_tickets: {n_fractional:,} non-whole-number value(s) "
                f"— column must be integer-compatible."
            )
        elif is_float_dtype:
            warnings.append(
                f"[Value range] support_tickets: dtype is float64 - "
                f"all {len(valid_tickets):,} values are whole numbers but column should be int."
            )

    # -- no future signup_date ------------------------------------------------
    if "signup_date" in df.columns:
        parsed_dates = pd.to_datetime(df["signup_date"], errors="coerce")
        # Count values that were non-null in raw but could not be parsed
        n_unparseable = int(parsed_dates.isna().sum() - df["signup_date"].isna().sum())
        n_future = int((parsed_dates > pd.Timestamp(today)).sum())
        issues["future_signup_dates"] = n_future
        issues["unparseable_signup_dates"] = n_unparseable

        if n_unparseable > 0:
            warnings.append(
                f"[Value range] signup_date: {n_unparseable:,} value(s) could not be parsed as dates."
            )
        if n_future > 0:
            failures.append(
                f"[Value range] signup_date: {n_future:,} date(s) are in the future."
            )

    stats["value_range_issues"] = issues


def _check_target_distribution(
    df: pd.DataFrame,
    failures: list[str],
    warnings: list[str],
    stats: dict[str, Any],
) -> None:
    """Check 5 — churn target has exactly 2 classes, each with >= 5 % of rows.

    Critical if fewer than 2 classes exist.
    Warning if any class represents < 5 % of rows.

    Args:
        df: DataFrame to validate.
        failures: Accumulator for critical error messages.
        warnings: Accumulator for non-critical concern messages.
        stats: Accumulator for counts and metadata.
    """
    if "churn" not in df.columns:
        stats["churn_distribution"] = {}
        return

    dist = df["churn"].dropna().value_counts(normalize=True).sort_index()
    stats["churn_distribution"] = {int(k): round(v, 4) for k, v in dist.items()}

    if len(dist) < 2:
        failures.append(
            f"[Target] churn has only {len(dist)} unique class(es) — at least 2 required."
        )
        return

    for cls, rate in dist.items():
        if rate < 0.05:
            warnings.append(
                f"[Target] churn class {int(cls)}: {rate:.1%} of rows — below 5 % minimum."
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_data_quality(df: pd.DataFrame) -> dict[str, Any]:
    """Run five data-quality checks on a customer DataFrame.

    Checks performed (in order):

    1. **Schema** — required columns present; numeric columns have numeric dtypes.
    2. **Row count** — at least 100 rows; warn if fewer than 1,000.
    3. **Null rates** — no column above 50 % nulls; warn above 20 %.
    4. **Value ranges** — non-negative ``total_spent``; ``return_rate`` in [0, 1];
       integer-compatible ``support_tickets``; no future ``signup_date``.
    5. **Target** — ``churn`` has exactly 2 classes, each >= 5 %.

    Args:
        df: DataFrame to validate (typically loaded from data/raw/customers.csv).

    Returns:
        Dictionary with keys:

        - ``success``    (*bool*): True only when *failures* is empty.
        - ``failures``   (*list[str]*): Critical errors that block model training.
        - ``warnings``   (*list[str]*): Non-blocking concerns to investigate.
        - ``statistics`` (*dict*): Counts and rates collected during the checks.
    """
    failures: list[str] = []
    warnings: list[str] = []
    stats: dict[str, Any] = {"column_count": len(df.columns)}

    _check_schema(df, failures, warnings, stats)
    _check_row_count(df, failures, warnings, stats)
    _check_null_rates(df, failures, warnings, stats)
    _check_value_ranges(df, failures, warnings, stats)
    _check_target_distribution(df, failures, warnings, stats)

    return {
        "success": len(failures) == 0,
        "failures": failures,
        "warnings": warnings,
        "statistics": stats,
    }


# ---------------------------------------------------------------------------
# Reporting helper
# ---------------------------------------------------------------------------


def _print_report(result: dict[str, Any], source: Path) -> None:
    """Print a human-readable pass/fail report to stdout.

    Args:
        result: Return value of :func:`check_data_quality`.
        source: Path to the CSV that was loaded, shown in the header.
    """
    status = "PASS" if result["success"] else "FAIL"
    sep = "=" * 54

    print(f"\n{sep}")
    print(f"  Data Quality Gate: {status}")
    print(f"  Source: {source}")
    print(sep)

    failures = result["failures"]
    warns = result["warnings"]
    stats = result["statistics"]

    if not failures and not warns:
        print("\n  All checks passed with no issues.")
    else:
        if failures:
            print(f"\nCRITICAL FAILURES ({len(failures)})")
            print("-" * 54)
            for msg in failures:
                print(f"  [FAIL]  {msg}")

        if warns:
            print(f"\nWARNINGS ({len(warns)})")
            print("-" * 54)
            for msg in warns:
                print(f"  [WARN]  {msg}")

    print(f"\nSTATISTICS")
    print("-" * 54)
    print(f"  Rows                    : {stats.get('row_count', '?'):>10,}")
    print(f"  Columns                 : {stats.get('column_count', '?'):>10}")

    churn_dist = stats.get("churn_distribution", {})
    if churn_dist:
        for cls, rate in sorted(churn_dist.items()):
            label = f"  Churn class {cls} rate"
            print(f"  {label:<22} : {rate:>9.1%}")

    null_rates = stats.get("null_rates", {})
    print(f"  Columns with nulls      : {len(null_rates):>10}")
    for col, rate in null_rates.items():
        print(f"    {col:<22} : {rate:>9.1%}")

    range_issues = stats.get("value_range_issues", {})
    if range_issues:
        print(f"  Value range checks")
        for key, val in range_issues.items():
            label = key.replace("_", " ").capitalize()
            print(f"    {label:<22} : {val!s:>9}")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _RAW_CSV.exists():
        print(f"ERROR: CSV not found at '{_RAW_CSV}'. Run 'python data/generate.py' first.")
        sys.exit(1)

    _df = pd.read_csv(_RAW_CSV)
    _result = check_data_quality(_df)
    _print_report(_result, _RAW_CSV)

    sys.exit(0 if _result["success"] else 1)
