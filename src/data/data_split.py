# src/data/data_split.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def split_data(
    df, target_column="result", test_size=0.2, val_size=0.25, random_state=42
):
    """
    Split data into train, validation, and test sets

    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of the target column
        test_size (float): Proportion of data to use for testing
        val_size (float): Proportion of training data to use for validation
        random_state (int): Random state for reproducibility

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info("Splitting data into train, validation, and test sets")

    # Ensure target column exists
    if target_column not in df.columns:
        logger.warning(f"Target column '{target_column}' not found in dataframe")
        if "Result" in df.columns:
            logger.info(f"Using 'Result' as target column instead")
            target_column = "Result"
        else:
            raise ValueError(f"Target column '{target_column}' not found in dataframe")

    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # First split: training + validation and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_val,
    )

    logger.info(
        f"Data split complete. Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
