# tests/conftest.py
import os
import pickle
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import application
from src.api.main import app


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app"""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_data():
    """Create sample insurance data for testing"""
    return pd.DataFrame(
        {
            "Gender": ["Male", "Female"],
            "Age": [35.0, 45.0],
            "HasDrivingLicense": [1, 1],
            "RegionID": [28, 10],
            "VehicleAge": ["1-2 Year", "< 1 Year"],
            "PastAccident": ["No", "Yes"],
            "AnnualPremium": ["£2,500", "£3,500"],
            "SalesChannelID": [26, 152],
            "DaysSinceCreated": [80, 120],
        }
    )


@pytest.fixture
def processed_sample_data(sample_data):
    """Create pre-processed sample data"""
    from src.data.preprocess import preprocess_data

    return preprocess_data(sample_data)


# In tests/conftest.py, update the mock_model_path fixture:
@pytest.fixture
def mock_model_path(tmpdir):
    """Create a mock model file for testing"""
    from sklearn.ensemble import RandomForestClassifier

    # Create a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)

    # Create dummy data that matches our feature pipeline output
    # Create with the right number of features (14)
    X = np.random.rand(100, 14)
    y = np.random.randint(0, 2, 100)

    # Train model
    model.fit(X, y)

    # Save model
    model_path = os.path.join(tmpdir, "test_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Create info file
    info_path = os.path.join(tmpdir, "model_info.txt")
    with open(info_path, "w") as f:
        f.write("model_type: RandomForestClassifier\n")
        f.write("model_version: test\n")

    # Set environment variable for model path
    os.environ["MODEL_PATH"] = model_path

    return model_path


@pytest.fixture
def prediction_payload():
    """Sample payload for prediction endpoint"""
    return {
        "gender": "Male",
        "age": 35.0,
        "has_driving_license": True,
        "region_id": 28,
        "vehicle_age": "1-2 Year",
        "past_accident": "No",
        "annual_premium": 2630.0,
        "sales_channel_id": 26,
        "days_since_created": 80,
    }


@pytest.fixture
def batch_prediction_payload(prediction_payload):
    """Sample payload for batch prediction endpoint"""
    return {
        "records": [
            prediction_payload,
            {
                "gender": "Female",
                "age": 45.0,
                "has_driving_license": True,
                "region_id": 10,
                "vehicle_age": "< 1 Year",
                "past_accident": "Yes",
                "annual_premium": 3500.0,
                "sales_channel_id": 152,
                "days_since_created": 120,
            },
        ]
    }
