# src/models/cross_validation.py
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def cross_validate_model(model, X, y, cv=5, scoring="roc_auc", random_state=42):
    """
    Perform cross-validation on a model

    Args:
        model: Model to validate
        X: Feature matrix
        y: Target vector
        cv (int): Number of folds
        scoring (str): Scoring metric
        random_state (int): Random state for reproducibility

    Returns:
        dict: Cross-validation results
    """
    logger.info(f"Performing {cv}-fold cross-validation with {scoring} scoring")

    # Create stratified k-fold
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    # Initialize metrics for each fold
    fold_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
    }

    # Perform cross-validation
    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # Train model
        model.fit(X_train_fold, y_train_fold)

        # Make predictions
        y_pred = model.predict(X_val_fold)
        y_pred_proba = model.predict_proba(X_val_fold)[:, 1]

        # Calculate metrics
        fold_metrics["accuracy"].append(accuracy_score(y_val_fold, y_pred))
        fold_metrics["precision"].append(
            precision_score(y_val_fold, y_pred, zero_division=0)
        )
        fold_metrics["recall"].append(recall_score(y_val_fold, y_pred, zero_division=0))
        fold_metrics["f1"].append(f1_score(y_val_fold, y_pred, zero_division=0))
        fold_metrics["roc_auc"].append(roc_auc_score(y_val_fold, y_pred_proba))

        # Log metrics for this fold
        logger.info(
            f"Fold {i+1}/{cv} - Accuracy: {fold_metrics['accuracy'][-1]:.4f}, "
            f"ROC-AUC: {fold_metrics['roc_auc'][-1]:.4f}"
        )

        # Log to MLflow if active run exists
        if mlflow.active_run():
            mlflow.log_metrics(
                {
                    f"fold_{i+1}_accuracy": fold_metrics["accuracy"][-1],
                    f"fold_{i+1}_precision": fold_metrics["precision"][-1],
                    f"fold_{i+1}_recall": fold_metrics["recall"][-1],
                    f"fold_{i+1}_f1": fold_metrics["f1"][-1],
                    f"fold_{i+1}_roc_auc": fold_metrics["roc_auc"][-1],
                }
            )

    # Calculate mean and std for each metric
    cv_results = {}
    for metric in fold_metrics:
        cv_results[f"{metric}_mean"] = np.mean(fold_metrics[metric])
        cv_results[f"{metric}_std"] = np.std(fold_metrics[metric])

        logger.info(
            f"CV {metric}: {cv_results[f'{metric}_mean']:.4f} ± {cv_results[f'{metric}_std']:.4f}"
        )

        # Log to MLflow if active run exists
        if mlflow.active_run():
            mlflow.log_metric(f"cv_{metric}_mean", cv_results[f"{metric}_mean"])
            mlflow.log_metric(f"cv_{metric}_std", cv_results[f"{metric}_std"])

    return cv_results
