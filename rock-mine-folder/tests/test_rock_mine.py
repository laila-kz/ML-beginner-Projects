"""Tests for the rock-vs-mine module."""

from pathlib import Path

import pandas as pd

from src.rock_mine import load_dataset, prepare_features


def test_load_dataset_reads_csv():
    """The sonar dataset should load from the data directory as a DataFrame."""
    data_path = Path(__file__).resolve().parents[1] / "data" / "sonar_data.csv"
    df = load_dataset(data_path)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_prepare_features_returns_feature_matrix_and_target():
    """The feature matrix and target vector should have consistent row counts."""
    data_path = Path(__file__).resolve().parents[1] / "data" / "sonar_data.csv"
    df = load_dataset(data_path)
    features, target = prepare_features(df)
    assert features.shape[0] == target.shape[0]
    assert features.shape[1] == df.shape[1] - 1
