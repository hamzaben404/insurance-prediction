# tests/unit/data/test_preprocess.py
import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (clean_column_names, clean_currency_values,
                                 preprocess_data)


def test_clean_column_names(sample_data):
    """Test column name cleaning"""
    # Original column names
    orig_cols = sample_data.columns.tolist()

    # Apply function
    result = clean_column_names(sample_data)

    # Check results
    assert len(result.columns) == len(orig_cols)
    assert all(col.lower() == col for col in result.columns)
    assert all(" " not in col for col in result.columns)


def test_clean_currency_values():
    """Test currency value cleaning"""
    # Test data
    df = pd.DataFrame({"annual_premium": ["£1,000", "£2,000.50", "£3,456.78"]})

    # Apply function
    result = clean_currency_values(df)

    # Check results
    assert "annual_premium" in result.columns
    assert result["annual_premium"].dtype == np.float64
    assert result["annual_premium"].tolist() == [1000.0, 2000.50, 3456.78]


def test_preprocess_data(sample_data):
    """Test full preprocessing pipeline"""
    # Apply function
    result = preprocess_data(sample_data)

    # Check column renaming
    assert "gender" in result.columns
    assert "age" in result.columns
    assert "annual_premium" in result.columns

    # Check currency conversion
    assert result["annual_premium"].dtype == np.float64
    assert len(result) == len(sample_data)
