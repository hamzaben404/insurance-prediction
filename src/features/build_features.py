# src/features/build_features.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def encode_categorical_features(df):
    """
    Encode categorical features using one-hot encoding with clean column names

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with encoded categorical features
    """
    logger.info("Encoding categorical features")

    # Create a copy to avoid modifying the original dataframe
    df_encoded = df.copy()

    # Define categorical columns
    categorical_cols = ["gender", "vehicle_age", "past_accident"]

    for col in categorical_cols:
        if col in df.columns:
            # Get dummies
            dummies = pd.get_dummies(df[col], prefix=col)

            # Clean up column names - replace spaces and special chars
            clean_cols = {}
            for dummy_col in dummies.columns:
                # Replace problematic characters with underscores or descriptive text
                clean_col = dummy_col.replace(" ", "_")
                clean_col = clean_col.replace("<", "less_than")
                clean_col = clean_col.replace(">", "more_than")
                clean_col = clean_col.replace("-", "to")
                clean_cols[dummy_col] = clean_col

            # Rename columns
            dummies = dummies.rename(columns=clean_cols)

            # Join with main dataframe
            df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)

            logger.info(
                f"Encoded {col} with clean column names: {list(dummies.columns)}"
            )

    logger.info(f"Encoded categorical features. New shape: {df_encoded.shape}")
    return df_encoded


def normalize_numerical_features(df):
    """
    Normalize numerical features

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with normalized numerical features
    """
    logger.info("Normalizing numerical features")

    # Define numerical columns to normalize
    numerical_cols = ["age", "annual_premium", "days_since_created"]

    # Create a copy to avoid modifying the original
    df_normalized = df.copy()

    # Apply standard scaling to each numerical column
    scaler = StandardScaler()
    for col in numerical_cols:
        if col in df.columns:
            # Reshape for scaler
            values = df[[col]].values
            df_normalized[col] = scaler.fit_transform(values)

    logger.info("Normalized numerical features")
    return df_normalized


def create_interaction_features(df):
    """
    Create interaction features between selected variables

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with additional interaction features
    """
    logger.info("Creating interaction features")

    # Create a copy to avoid modifying the original
    df_interactions = df.copy()

    # Age and annual premium interaction
    if "age" in df.columns and "annual_premium" in df.columns:
        # Avoid division by zero
        df_interactions["age_premium_ratio"] = df["age"] / df["annual_premium"].replace(
            0, 0.001
        )

    # Age and days since created interaction
    if "age" in df.columns and "days_since_created" in df.columns:
        df_interactions["age_days_ratio"] = (
            df["age"] * 365 / df["days_since_created"].replace(0, 1)
        )

    logger.info(f"Created interaction features. New shape: {df_interactions.shape}")
    return df_interactions


def create_feature_pipeline(df):
    """
    Run full feature engineering pipeline

    Args:
        df (pd.DataFrame): Clean preprocessed dataframe

    Returns:
        pd.DataFrame: Feature-engineered dataframe ready for modeling
    """
    logger.info("Starting feature engineering pipeline")

    # Apply feature engineering steps
    df = encode_categorical_features(df)
    df = normalize_numerical_features(df)
    df = create_interaction_features(df)

    logger.info("Feature engineering pipeline completed")
    return df
