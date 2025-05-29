# src/data/profile_data.py
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def create_data_profile(df, output_dir="reports"):
    """
    Create a data profile report

    Args:
        df (pd.DataFrame): Input dataframe
        output_dir (str): Directory to save profiles

    Returns:
        dict: Data profile statistics
    """
    logger.info("Creating data profile")

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Basic statistics
    profile = {
        "dataset_info": {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "memory_usage": df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
        },
        "missing_values": df.isnull().sum().to_dict(),
        "column_stats": {},
    }

    # Per-column statistics
    for col in df.columns:
        col_type = str(df[col].dtype)

        # Base statistics for all columns
        col_stats = {
            "dtype": col_type,
            "unique_values": df[col].nunique(),
            "missing_values": df[col].isnull().sum(),
            "missing_percentage": round(df[col].isnull().sum() / len(df) * 100, 2),
        }

        # Additional statistics for numerical columns
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats.update(
                {
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "mean": df[col].mean(),
                    "median": df[col].median(),
                    "std": df[col].std(),
                }
            )
        # Additional statistics for categorical/object columns
        else:
            # Get value counts for categorical columns (top 10)
            value_counts = df[col].value_counts().head(10).to_dict()
            col_stats["value_counts"] = value_counts

        profile["column_stats"][col] = col_stats

    # Save profile to JSON
    with open(f"{output_dir}/data_profile.json", "w") as f:
        json.dump(profile, f, indent=4, default=str)

    logger.info(f"Data profile saved to {output_dir}/data_profile.json")

    # Create visualizations
    create_profile_visualizations(df, output_dir)

    return profile


def create_profile_visualizations(df, output_dir="reports"):
    """
    Create visualizations for data profile

    Args:
        df (pd.DataFrame): Input dataframe
        output_dir (str): Directory to save visualizations
    """
    logger.info("Creating data profile visualizations")

    # Create visualizations directory
    viz_dir = Path(output_dir) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # 1. Missing values heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/missing_values_heatmap.png")
    plt.close()

    # 2. Distribution of numerical features
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numerical_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/dist_{col}.png")
        plt.close()

    # 3. Count plots for categorical features
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        if df[col].nunique() < 15:  # Only plot if not too many categories
            plt.figure(figsize=(10, 5))
            sns.countplot(y=col, data=df, order=df[col].value_counts().index)
            plt.title(f"Count plot of {col}")
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/count_{col}.png")
            plt.close()

    # 4. If target variable exists, show its distribution
    if "result" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x="result", data=df)
        plt.title("Distribution of Target Variable")
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/target_distribution.png")
        plt.close()

        # 5. Create correlation heatmap
        plt.figure(figsize=(12, 10))
        corr_df = df.select_dtypes(include=["int64", "float64"]).corr()
        sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/correlation_heatmap.png")
        plt.close()

        # 6. Feature relationship with target
        for col in numerical_cols:
            if col != "result":
                plt.figure(figsize=(8, 4))
                sns.boxplot(x="result", y=col, data=df)
                plt.title(f"{col} by Target")
                plt.tight_layout()
                plt.savefig(f"{viz_dir}/target_by_{col}.png")
                plt.close()

    logger.info(f"Visualizations saved to {viz_dir}")
