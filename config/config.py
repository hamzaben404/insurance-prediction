import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
load_dotenv()

# Project base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model paths
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# MLflow settings
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Model settings
MODEL_PATH = os.getenv("MODEL_PATH", str(MODELS_DIR / "comparison/production/production_model.pkl"))

# Feature engineering configuration (for consistency)
NUMERICAL_FEATURES = ["Age", "DaysSinceCreated"]
CATEGORICAL_FEATURES = ["Gender", "VehicleAge", "PastAccident"]
TARGET_COLUMN = "Result"

# Model training configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2
