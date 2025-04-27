# src/models/model_selection.py
import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns

from src.models.mlflow_utils import (log_dataset_info, log_metrics,
                                     log_model_params, register_model,
                                     setup_mlflow, start_run)
from src.models.model_factory import get_available_models
from src.models.train_model import train_model
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def train_multiple_models(
    X_train,
    y_train,
    X_val,
    y_val,
    models_to_try=None,
    base_output_dir="models/comparison",
    tracking_uri=None,
):
    """
    Train multiple models and compare their performance

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        models_to_try (list): List of model types to try
        base_output_dir (str): Base directory for outputs
        tracking_uri (str): MLflow tracking URI

    Returns:
        pd.DataFrame: Comparison of model performances
    """
    # Setup MLflow
    experiment_id = setup_mlflow("insurance_model_selection", tracking_uri)

    # Use all available models if none specified
    if models_to_try is None:
        models_to_try = get_available_models()

    # Create base output directory
    os.makedirs(base_output_dir, exist_ok=True)

    # Results container
    model_results = []

    # Train each model
    for model_type in models_to_try:
        logger.info(f"Training {model_type} model")

        # Create output directory for this model
        model_dir = os.path.join(base_output_dir, model_type)
        os.makedirs(model_dir, exist_ok=True)

        # Start MLflow run
        with start_run(run_name=f"model_selection_{model_type}"):
            # Log dataset info
            log_dataset_info(
                {
                    "train_shape": X_train.shape,
                    "val_shape": X_val.shape,
                    "features": (
                        X_train.columns.tolist() if hasattr(X_train, "columns") else []
                    ),
                    "target": "insurance_purchase",
                }
            )

            # Train model
            try:
                results = train_model(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    model_type=model_type,
                    output_dir=model_dir,
                    do_cv=True,
                    cv=5,
                    tune_hyperparams=False,  # For quick comparison
                )

                # Add to results
                model_results.append(
                    {
                        "model_type": model_type,
                        "train_accuracy": results["train_metrics"]["accuracy"],
                        "train_precision": results["train_metrics"]["precision"],
                        "train_recall": results["train_metrics"]["recall"],
                        "train_f1": results["train_metrics"]["f1"],
                        "train_roc_auc": results["train_metrics"].get(
                            "roc_auc", np.nan
                        ),
                        "val_accuracy": results["val_metrics"]["accuracy"],
                        "val_precision": results["val_metrics"]["precision"],
                        "val_recall": results["val_metrics"]["recall"],
                        "val_f1": results["val_metrics"]["f1"],
                        "val_roc_auc": results["val_metrics"].get("roc_auc", np.nan),
                        "model_path": results["model_path"],
                    }
                )

                # Log metrics to MLflow
                log_metrics(
                    {
                        "val_accuracy": results["val_metrics"]["accuracy"],
                        "val_precision": results["val_metrics"]["precision"],
                        "val_recall": results["val_metrics"]["recall"],
                        "val_f1": results["val_metrics"]["f1"],
                    }
                )

                if "roc_auc" in results["val_metrics"]:
                    log_metrics({"val_roc_auc": results["val_metrics"]["roc_auc"]})

                # Log model to MLflow
                mlflow.sklearn.log_model(results["model"], "model")

            except Exception as e:
                logger.error(f"Error training {model_type} model: {e}")

    # Create comparison dataframe
    comparison_df = pd.DataFrame(model_results)

    # Save comparison
    comparison_path = os.path.join(base_output_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)

    # Plot comparison
    plt.figure(figsize=(12, 8))
    comparison_metrics = comparison_df[
        [
            "model_type",
            "val_accuracy",
            "val_precision",
            "val_recall",
            "val_f1",
            "val_roc_auc",
        ]
    ]
    comparison_metrics = comparison_metrics.melt(
        id_vars=["model_type"], var_name="Metric", value_name="Value"
    )

    sns.barplot(x="model_type", y="Value", hue="Metric", data=comparison_metrics)
    plt.title("Model Comparison")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()

    comparison_plot_path = os.path.join(base_output_dir, "model_comparison.png")
    plt.savefig(comparison_plot_path)

    return comparison_df


def select_best_model(comparison_df, metric="val_f1", register=True):
    """
    Select the best model based on a metric

    Args:
        comparison_df (pd.DataFrame): Model comparison dataframe
        metric (str): Metric to use for selection
        register (bool): Whether to register the model

    Returns:
        tuple: (best_model_type, best_model_path)
    """
    logger.info(f"Selecting best model based on {metric}")

    # Get best model
    best_idx = comparison_df[metric].idxmax()
    best_row = comparison_df.iloc[best_idx]

    best_model_type = best_row["model_type"]
    best_model_path = best_row["model_path"]
    best_metric_value = best_row[metric]

    logger.info(
        f"Best model: {best_model_type} with {metric} = {best_metric_value:.4f}"
    )

    # Register best model if requested
    if register and mlflow.active_run() is not None:
        try:
            # Load best model
            import pickle

            # with open(best_model_path, "rb") as f:
            #     best_model = pickle.load(f)
            import joblib

            best_model = joblib.load(best_model_path)

            # Log and register
            mlflow.sklearn.log_model(best_model, "best_model")
            register_model(best_model, "insurance_prediction_model", "Production")

            logger.info(f"Registered {best_model_type} as production model")
        except Exception as e:
            logger.error(f"Error registering model: {e}")

    return best_model_type, best_model_path
