# tests/end_to_end_test.py
import os
import sys

import numpy as np
import pandas as pd
import pytest

from src.data.data_split import split_data
from src.data.load_data import load_dataset
from src.data.preprocess import preprocess_data
from src.data.validate_data import validate_dataset
from src.features.build_features import create_feature_pipeline

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    n_samples = 1000

    data = pd.DataFrame(
        {
            "id": range(1, n_samples + 1),
            "Gender": np.random.choice(["Male", "Female"], n_samples),
            "Age": np.random.randint(18, 70, n_samples),
            "HasDrivingLicense": np.random.choice([0, 1], n_samples),
            "RegionID": np.random.randint(1, 50, n_samples),
            "VehicleAge": np.random.choice(["< 1 Year", "1-2 Year", "> 2 Years"], n_samples),
            "PastAccident": np.random.choice(["Yes", "No"], n_samples),
            "AnnualPremium": ["£" + str(np.random.randint(1000, 5000)) for _ in range(n_samples)],
            "SalesChannelID": np.random.randint(1, 200, n_samples),
            "DaysSinceCreated": np.random.randint(1, 365, n_samples),
            "Result": np.random.choice([0, 1], n_samples),
        }
    )

    return data


@pytest.fixture
def temp_data_file(tmp_path, sample_data):
    """Create a temporary CSV file with sample data"""
    file_path = tmp_path / "test_train.csv"
    sample_data.to_csv(file_path, index=False)
    return str(file_path)


@pytest.fixture
def output_dir(tmp_path):
    """Create a temporary output directory"""
    output_path = tmp_path / "test_output"
    output_path.mkdir(exist_ok=True)
    return str(output_path)


class TestEndToEndPipeline:
    """End-to-end tests for the complete ML pipeline"""

    def test_data_loading(self, temp_data_file, sample_data):
        """Test data loading functionality"""
        df = load_dataset(temp_data_file)
        assert df is not None
        assert df.shape == sample_data.shape
        assert list(df.columns) == list(sample_data.columns)

    def test_data_validation(self, sample_data):
        """Test data validation"""
        valid, results = validate_dataset(sample_data)
        assert isinstance(valid, bool)
        assert isinstance(results, dict)
        assert "checks" in results
        # The validation should pass for our generated data
        assert valid is True

    def test_data_preprocessing(self, sample_data):
        """Test data preprocessing"""
        df_processed = preprocess_data(sample_data)
        assert df_processed is not None
        assert len(df_processed) == len(sample_data)
        # Check that AnnualPremium is now numeric
        assert pd.api.types.is_numeric_dtype(df_processed["AnnualPremium"])
        # Check column name standardization
        assert "annual_premium" in df_processed.columns

    def test_feature_engineering(self, sample_data):
        """Test feature engineering pipeline"""
        df_processed = preprocess_data(sample_data)
        df_featured = create_feature_pipeline(df_processed)

        assert df_featured is not None
        assert len(df_featured) == len(sample_data)
        # Should have more columns after feature engineering
        assert len(df_featured.columns) > len(df_processed.columns)

    def test_data_splitting(self, sample_data):
        """Test data splitting"""
        df_processed = preprocess_data(sample_data)
        df_featured = create_feature_pipeline(df_processed)

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_featured)

        # Check shapes
        assert len(X_train) + len(X_val) + len(X_test) == len(df_featured)
        assert len(y_train) == len(X_train)
        assert len(y_val) == len(X_val)
        assert len(y_test) == len(X_test)

        # Check split proportions (60/20/20)
        total_samples = len(df_featured)
        assert 0.55 < len(X_train) / total_samples < 0.65
        assert 0.15 < len(X_val) / total_samples < 0.25
        assert 0.15 < len(X_test) / total_samples < 0.25

    def test_complete_pipeline(self, temp_data_file, output_dir):
        """Test the complete pipeline from loading to saving"""
        # Load data
        df = load_dataset(temp_data_file)
        assert df is not None

        # Validate data
        valid, results = validate_dataset(df)
        assert valid is True

        # Preprocess data
        df_processed = preprocess_data(df)
        assert df_processed is not None

        # Feature engineering
        df_featured = create_feature_pipeline(df_processed)
        assert df_featured is not None

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_featured)
        assert X_train is not None

        # Save output
        output_file = os.path.join(output_dir, "featured_data.csv")
        df_featured.to_csv(output_file, index=False)
        assert os.path.exists(output_file)

        # Verify saved file
        df_loaded = pd.read_csv(output_file)
        assert df_loaded.shape == df_featured.shape


@pytest.mark.skipif(
    not os.path.exists("data/raw/train.csv"), reason="Actual train.csv file not available"
)
class TestWithRealData:
    """Tests using the actual train.csv file (if available)"""

    def test_real_data_pipeline(self):
        """Test pipeline with real data"""
        train_path = "data/raw/train.csv"
        output_path = "data/processed/test_output"

        # Create output directory
        os.makedirs(output_path, exist_ok=True)

        # Load and process
        df = load_dataset(train_path)
        valid, _ = validate_dataset(df)
        assert valid is True

        df_processed = preprocess_data(df)
        df_featured = create_feature_pipeline(df_processed)

        # Save
        df_featured.to_csv(f"{output_path}/featured_data.csv", index=False)
        assert os.path.exists(f"{output_path}/featured_data.csv")
