"""
Data loading utilities for the RetailAI Churn Intelligence System.

Loads the raw customer CSV and prints diagnostic reports on shape,
column types, summary statistics, and missing values.

Usage:
    python src/data/loader.py
"""

from pathlib import Path

import pandas as pd

RAW_CSV = Path("data/raw/customers.csv")

# Column widths for aligned console output
_COL_W = 26
_SEP_NARROW = "-" * 44
_SEP_WIDE = "-" * 72


def load_customers(path: Path = RAW_CSV) -> pd.DataFrame:
    """Load the raw customer CSV into a DataFrame.

    Args:
        path: Path to the CSV file (default: data/raw/customers.csv).

    Returns:
        DataFrame containing all customer records.

    Raises:
        FileNotFoundError: If the CSV does not exist at the given path.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Customer CSV not found at '{path}'.\n"
            "Run 'python data/generate.py' first."
        )
    return pd.read_csv(path)


def report_shape(df: pd.DataFrame) -> None:
    """Print the number of rows and columns.

    Args:
        df: DataFrame to inspect.
    """
    rows, cols = df.shape
    print(f"Shape: {rows:,} rows x {cols} columns")


def report_columns(df: pd.DataFrame) -> None:
    """Print each column name alongside its dtype.

    Args:
        df: DataFrame to inspect.
    """
    print("\nColumns and data types")
    print(_SEP_NARROW)
    for col, dtype in df.dtypes.items():
        print(f"  {col:<{_COL_W}} {dtype}")


def report_statistics(df: pd.DataFrame) -> None:
    """Print mean, std, min, and max for every numeric column.

    Args:
        df: DataFrame to inspect.
    """
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        print("\nNo numeric columns found.")
        return

    stats = numeric.agg(["mean", "std", "min", "max"])
    print("\nSummary statistics (numeric columns)")
    print(_SEP_WIDE)
    print(stats.T.to_string(float_format=lambda x: f"{x:>14.4f}"))


def report_missing(df: pd.DataFrame) -> None:
    """Print missing-value counts and percentages for each column.

    Columns with zero missing values are omitted from the output.

    Args:
        df: DataFrame to inspect.
    """
    n = len(df)
    counts = df.isna().sum()
    missing = counts[counts > 0].sort_values(ascending=False)

    print("\nMissing values")
    print(_SEP_NARROW)
    if missing.empty:
        print("  None detected.")
        return

    for col, count in missing.items():
        pct = count / n * 100
        print(f"  {col:<{_COL_W}} {count:>6,}  ({pct:5.2f}%)")


def run_diagnostics(path: Path = RAW_CSV) -> pd.DataFrame:
    """Run all diagnostic reports for the customer dataset.

    Loads the CSV from *path*, then sequentially prints shape,
    column types, summary statistics, and missing-value counts.

    Args:
        path: Path to the CSV file (default: data/raw/customers.csv).

    Returns:
        The loaded DataFrame.
    """
    print("=" * 44)
    print("  RetailAI - Customer data diagnostics")
    print("=" * 44)
    print(f"Source: {path}\n")

    df = load_customers(path)

    report_shape(df)
    report_columns(df)
    report_statistics(df)
    report_missing(df)

    print()
    return df


if __name__ == "__main__":
    run_diagnostics()
