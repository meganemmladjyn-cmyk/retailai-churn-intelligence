"""Feature engineering tests for create_features()."""

import pandas as pd
import pytest

from src.features.engineering import create_features

_MIN_OUTPUT_COLS = 18


def test_create_features_expands_column_count(sample_cleaned_df: pd.DataFrame) -> None:
    n_input = sample_cleaned_df.shape[1]
    out = create_features(sample_cleaned_df)
    assert out.shape[1] > n_input, (
        f"create_features() must add columns (got {out.shape[1]} from {n_input})"
    )
    assert out.shape[1] >= _MIN_OUTPUT_COLS, (
        f"Expected at least {_MIN_OUTPUT_COLS} columns, got {out.shape[1]}"
    )


def test_create_features_no_nan_in_numeric_columns(sample_cleaned_df: pd.DataFrame) -> None:
    out = create_features(sample_cleaned_df)
    numeric_cols = out.select_dtypes(include="number").columns
    total_nulls = out[numeric_cols].isna().sum().sum()
    assert total_nulls == 0, (
        f"create_features() produced {total_nulls} NaN(s) in numeric columns"
    )


def test_is_recent_buyer_is_binary(sample_cleaned_df: pd.DataFrame) -> None:
    out = create_features(sample_cleaned_df)
    unexpected = set(out["is_recent_buyer"].unique()) - {0, 1}
    assert not unexpected, (
        f"is_recent_buyer contains unexpected values: {unexpected}"
    )


def test_recency_segment_values_are_valid(sample_cleaned_df: pd.DataFrame) -> None:
    out = create_features(sample_cleaned_df)
    unexpected = set(out["recency_segment"].unique()) - {0, 1, 2}
    assert not unexpected, (
        f"recency_segment contains unexpected values: {unexpected}"
    )
