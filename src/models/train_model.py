# src/models/train_model.py
import os
import pickle
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd

from src.models.cross_validation import cross_validate_model
from src.models.evaluation import (evaluate_model, evaluate_threshold,
                                   generate_classification_report)
from src.models.hyperparameter_tuning import (get_default_param_grids,
                                              grid_search_cv, random_search_cv)
from src.models.interpretation import (generate_shap_explanation,
                                       plot_feature_importance,
                                       plot_partial_dependence)
from src.models.mlflow_utils import (log_dataset_info, log_metrics,
                                     log_model_params, setup_mlflow, start_run)
from src.models.model_factory import create_model, get_available_models
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def train_model(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    model_type="random_forest",
    params=None,
    do_cv=True,
    cv=5,
    tune_hyperparams=False,
    tuning_method="grid",
    output_dir=None,
):
    """
    Train a model with the specified parameters

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features (optional)
        y_val: Validation target (optional)
        model_type (str): Type of model to train
        params (dict): Model parameters
        do_cv (bool): Whether to perform cross-validation
        cv (int): Number of cross-validation folds
        tune_hyperparams (bool): Whether to tune hyperparameters
        tuning_method (str): Hyperparameter tuning method ("grid" or "random")
        output_dir (str): Directory to save outputs

    Returns:
        dict: Training results including the trained model
    """
    if output_dir is None:
        output_dir = os.path.join(
            "models", f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Training {model_type} model, output directory: {output_dir}")

    # Create model
    model = create_model(model_type, params)

    # Training results
    results = {
        "model_type": model_type,
        "output_dir": output_dir,
        "train_shape": X_train.shape,
        "val_shape": X_val.shape if X_val is not None else None,
    }

    # Cross-validation
    if do_cv:
        logger.info("Performing cross-validation")
        cv_results = cross_validate_model(model, X_train, y_train, cv=cv)
        results["cv_results"] = cv_results

    # Hyperparameter tuning
    if tune_hyperparams:
        logger.info(f"Tuning hyperparameters using {tuning_method} search")

        # Get parameter grid/distributions
        param_grids = get_default_param_grids()

        if model_type in param_grids:
            param_grid = param_grids[model_type]

            # Perform tuning
            if tuning_method == "grid":
                tuned_model, best_params, best_score = grid_search_cv(
                    model, param_grid, X_train, y_train, cv=cv
                )
            else:  # random search
                tuned_model, best_params, best_score = random_search_cv(
                    model, param_grid, X_train, y_train, cv=cv, n_iter=10
                )

            # Update model and results
            model = tuned_model
            results["tuning_method"] = tuning_method
            results["best_params"] = best_params
            results["best_tuning_score"] = best_score
        else:
            logger.warning(
                f"No parameter grid defined for {model_type}, skipping tuning"
            )

    # Train final model
    logger.info("Training final model")
    model.fit(X_train, y_train)
    results["model"] = model

    # Evaluate on training data
    logger.info("Evaluating on training data")
    train_metrics = evaluate_model(
        model, X_train, y_train, output_dir=os.path.join(output_dir, "train_evaluation")
    )
    results["train_metrics"] = train_metrics

    # Evaluate on validation data if provided
    if X_val is not None and y_val is not None:
        logger.info("Evaluating on validation data")
        val_metrics = evaluate_model(
            model, X_val, y_val, output_dir=os.path.join(output_dir, "val_evaluation")
        )
        results["val_metrics"] = val_metrics

        # Generate detailed evaluation and interpretation
        evaluate_threshold(
            model, X_val, y_val, output_dir=os.path.join(output_dir, "val_evaluation")
        )
        generate_classification_report(
            model, X_val, y_val, output_dir=os.path.join(output_dir, "val_evaluation")
        )

    # Generate model interpretation
    logger.info("Generating model interpretation")
    interpretation_dir = os.path.join(output_dir, "interpretation")
    plot_feature_importance(model, X_train, output_dir=interpretation_dir)

    try:
        generate_shap_explanation(
            model,
            X_val if X_val is not None else X_train.sample(min(100, len(X_train))),
            output_dir=interpretation_dir,
        )
    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}")

    try:
        plot_partial_dependence(
            model,
            X_val if X_val is not None else X_train,
            output_dir=interpretation_dir,
        )
    except Exception as e:
        logger.warning(f"Partial dependence plots failed: {e}")

    # Save model
    model_path = os.path.join(output_dir, "model.pkl")
    # with open(model_path, "wb") as f:
    #     pickle.dump(model, f)

    import joblib

    joblib.dump(model, model_path)

    logger.info(f"Model saved to {model_path}")
    results["model_path"] = model_path

    return results
