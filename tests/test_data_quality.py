"""Data quality tests on the cleaned customer dataset."""

import pandas as pd
import pytest

from src.data.quality import check_data_quality

_KEY_COLUMNS = ["customer_id", "age", "total_orders", "total_spent", "churn"]


def test_no_nulls_in_key_columns(sample_cleaned_df: pd.DataFrame) -> None:
    for col in _KEY_COLUMNS:
        null_count = sample_cleaned_df[col].isna().sum()
        assert null_count == 0, f"Column '{col}' has {null_count} null(s) after cleaning"


def test_churn_rate_in_expected_range(sample_cleaned_df: pd.DataFrame) -> None:
    rate = sample_cleaned_df["churn"].mean()
    assert 0.15 <= rate <= 0.35, (
        f"Churn rate {rate:.2%} is outside expected range [15%, 35%]"
    )


def test_broken_df_missing_churn_column_fails_quality_gate() -> None:
    broken = pd.DataFrame({
        "customer_id": ["C001", "C002", "C003"],
        "age": [25, 30, 45],
        "total_orders": [5, 10, 3],
        "total_spent": [200.0, 400.0, 80.0],
    })
    result = check_data_quality(broken)
    assert result["success"] is False, (
        "Quality gate should fail when 'churn' column is missing"
    )
