"""Tests for the diabetes prediction module."""

from pathlib import Path

import pandas as pd

from src.diabetes_prediction import load_dataset, prepare_features


def test_load_dataset_reads_csv():
    """The dataset should load from the data directory as a DataFrame."""
    data_path = Path(__file__).resolve().parents[1] / "data" / "diabetes.csv"
    df = load_dataset(data_path)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_prepare_features_returns_arrays():
    """Feature extraction should separate X and y correctly."""
    data_path = Path(__file__).resolve().parents[1] / "data" / "diabetes.csv"
    df = load_dataset(data_path)
    features, target = prepare_features(df)
    assert features.shape[0] == target.shape[0]
    assert features.shape[1] == df.shape[1] - 1
