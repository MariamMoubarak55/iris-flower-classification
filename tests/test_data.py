"""Unit tests for data loading and preprocessing."""

import pytest
import pandas as pd
from src.main.data_loader import load_data
from src.main.preprocessing import split_data, FEATURE_COLUMNS, sample_stratified_split


def test_load_data_returns_dataframe():
    """Test that load_data returns non-empty DataFrame with expected columns."""
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "species" in df.columns
    for feature in FEATURE_COLUMNS:
        assert feature in df.columns


def test_split_data_shapes():
    """Test train_test_split proportions and stratification."""
    df = load_data()
    test_size = 0.2
    X_train, X_test, y_train, y_test = split_data(df, test_size=test_size, random_state=42)

    total_len = len(df)
    assert len(X_train) + len(X_test) == total_len
    assert len(y_train) + len(y_test) == total_len
    assert abs(len(X_test) / total_len - test_size) < 0.05


def test_sample_stratified_split():
    """Test custom stratified sampling from Slide 10."""
    df = load_data()
    n_samples = 5
    train_df, test_df = sample_stratified_split(df, n_samples=n_samples)

    # 3 classes * 5 samples = 15 test rows
    assert len(test_df) == 15
    assert len(train_df) == len(df) - 15
    assert test_df["species"].value_counts().tolist() == [5, 5, 5]
