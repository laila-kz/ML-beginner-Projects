"""Tests for the house price regression module."""

import pandas as pd

from src.house_price import load_dataset, split_features_target


def test_load_dataset_returns_dataframe():
    """The California housing dataset should load as a DataFrame."""
    df = load_dataset()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_split_features_target_returns_expected_shapes():
    """Feature and target extraction should align by row count."""
    df = load_dataset()
    features, target = split_features_target(df)
    assert features.shape[0] == target.shape[0]
    assert "price" not in features.columns
