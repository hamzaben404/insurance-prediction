# src/run_training_pipeline.py
import argparse
import os

import mlflow
import pandas as pd

from config.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.data.load_data import load_dataset
from src.models.model_selection import select_best_model, train_multiple_models
from src.utils.logging import setup_logger
from src.utils.mlflow_utils import setup_mlflow

logger = setup_logger(__name__)


def main(args):
    """
    Run the training pipeline

    Args:
        args: Command line arguments
    """
    logger.info("Starting training pipeline")

    # Setup MLflow
    setup_mlflow(args.experiment_name, args.tracking_uri)

    # Load data
    logger.info("Loading processed data")
    X_train = pd.read_csv(os.path.join(args.data_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(args.data_dir, "y_train.csv")).iloc[:, 0]
    X_val = pd.read_csv(os.path.join(args.data_dir, "X_val.csv"))
    y_val = pd.read_csv(os.path.join(args.data_dir, "y_val.csv")).iloc[:, 0]

    logger.info(
        f"Loaded data - X_train: {X_train.shape}, y_train: {y_train.shape}, X_val: {X_val.shape}, y_val: {y_val.shape}"
    )

    # Define models to try
    if args.models:
        models_to_try = args.models.split(",")
    else:
        # Default models - simpler set for faster execution
        models_to_try = ["logistic_regression", "random_forest", "xgboost"]

    logger.info(f"Training models: {models_to_try}")

    # Train and compare models
    comparison_df = train_multiple_models(
        X_train,
        y_train,
        X_val,
        y_val,
        models_to_try=models_to_try,
        base_output_dir=args.output_dir,
        tracking_uri=args.tracking_uri,
    )

    # Select best model
    best_model_type, best_model_path = select_best_model(
        comparison_df, metric=args.metric, register=True
    )

    logger.info(f"Training pipeline completed. Best model: {best_model_type}")
    logger.info(f"Best model saved at: {best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the training pipeline")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(PROCESSED_DATA_DIR, "train"),
        help="Directory containing processed data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(MODELS_DIR, "comparison"),
        help="Directory to save models and results",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to train",
    )
    parser.add_argument(
        "--metric", type=str, default="val_f1", help="Metric to use for model selection"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="insurance_prediction",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--tracking-uri", type=str, default=None, help="MLflow tracking URI"
    )

    args = parser.parse_args()

    main(args)
