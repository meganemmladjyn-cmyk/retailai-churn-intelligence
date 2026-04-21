"""Production model loading and inference tests."""

import pandas as pd
import pytest


def test_production_model_loads(production_model) -> None:
    assert production_model is not None


def test_predict_proba_values_in_unit_interval(production_model, model_input_df: pd.DataFrame) -> None:
    probas = production_model.predict_proba(model_input_df)
    assert probas.min() >= 0.0, f"predict_proba returned value below 0: {probas.min()}"
    assert probas.max() <= 1.0, f"predict_proba returned value above 1: {probas.max()}"


def test_prediction_output_shape_matches_input(production_model, model_input_df: pd.DataFrame) -> None:
    probas = production_model.predict_proba(model_input_df)
    assert probas.shape[0] == len(model_input_df), (
        f"Expected {len(model_input_df)} rows in output, got {probas.shape[0]}"
    )
    assert probas.shape[1] == 2, (
        f"Expected 2 probability columns (binary classification), got {probas.shape[1]}"
    )
