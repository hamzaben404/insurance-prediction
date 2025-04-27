# tests/unit/features/test_features.py
import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (create_feature_pipeline,
                                         encode_categorical_features,
                                         normalize_numerical_features)


# tests/unit/features/test_features.py
def test_encode_categorical_features(processed_sample_data):
    """Test categorical feature encoding"""
    # Apply function
    result = encode_categorical_features(processed_sample_data)

    # Check encoding - use correct capitalization
    assert "gender_Male" in result.columns or "gender_Female" in result.columns
    assert "gender" not in result.columns  # Original column should be dropped

    # Check shape - should have more columns after one-hot encoding
    assert result.shape[1] >= processed_sample_data.shape[1]


def test_normalize_numerical_features(processed_sample_data):
    """Test numerical feature normalization"""
    # Apply function
    result = normalize_numerical_features(processed_sample_data)

    # Check normalization of age
    if "age" in result.columns:
        assert -3 < result["age"].mean() < 3  # Should be roughly centered

    # Check normalization of annual_premium
    if "annual_premium" in result.columns:
        assert -3 < result["annual_premium"].mean() < 3


def test_feature_pipeline(processed_sample_data):
    """Test full feature engineering pipeline"""
    # Apply function
    result = create_feature_pipeline(processed_sample_data)

    # Print the columns to help with debugging
    print(f"Input columns: {processed_sample_data.columns.tolist()}")
    print(f"Output columns: {result.columns.tolist()}")
    print(f"Feature count: {result.shape[1]}")

    # Check overall shape - should have more columns after feature engineering
    assert result.shape[0] == processed_sample_data.shape[0]  # Same number of rows
    assert result.shape[1] >= processed_sample_data.shape[1]  # More columns

    # Check for categorical encodings
    assert any(col.startswith("gender_") for col in result.columns)
    assert any(col.startswith("vehicle_age_") for col in result.columns)

    # Check for interaction features
    assert any("ratio" in col for col in result.columns)
