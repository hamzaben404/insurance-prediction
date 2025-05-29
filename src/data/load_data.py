# src/data/load_data.py
import pandas as pd

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def load_dataset(file_path):
    """
    Load dataset from CSV file

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded dataframe
    """
    logger.info(f"Loading data from {file_path}")
    try:
        # Handle currency values and other potential parsing issues
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def get_data_info(df):
    """
    Get basic information about the dataset

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        dict: Dictionary with basic dataset information
    """
    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "unique_values": {col: df[col].nunique() for col in df.columns},
    }
    return info
