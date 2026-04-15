import pytest

from ml.features import FEATURE_COLUMNS, TARGET_COLUMN


def test_feature_columns_not_empty():
    assert len(FEATURE_COLUMNS) > 0


def test_feature_columns_no_duplicates():
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


def test_target_column():
    assert TARGET_COLUMN == "is_churned"


def test_generate_customer():
    from data.generate import generate_customer

    customer = generate_customer(1)

    assert customer["customer_id"] == "RET-000001"
    assert 18 <= customer["age"] <= 75
    assert 0.0 <= customer["return_rate"] <= 1.0
    assert 0.0 <= customer["email_open_rate"] <= 1.0
    assert customer["is_churned"] in (0, 1)
    assert customer["total_orders"] >= 1


def test_feature_columns_in_generated_data():
    from data.generate import generate_customer

    customer = generate_customer(42)
    for col in FEATURE_COLUMNS:
        assert col in customer, f"Feature column '{col}' missing from generated data"
