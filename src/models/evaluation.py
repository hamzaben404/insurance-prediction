# src/models/evaluation.py
import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def evaluate_model(model, X, y, threshold=0.5, output_dir=None):
    """
    Evaluate model performance on a dataset

    Args:
        model: Trained model
        X: Feature matrix
        y: True labels
        threshold (float): Probability threshold for classification
        output_dir (str): Directory to save outputs

    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating model performance")

    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Make predictions
    try:
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
    except:
        logger.warning("Model doesn't support predict_proba, using predict instead")
        y_pred = model.predict(X)
        y_prob = y_pred  # Not actual probabilities

    # Calculate basic metrics
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
    }

    # AUC if probabilities are available
    if np.array_equal(y_prob, y_pred):
        logger.warning("ROC-AUC not calculated - no probability predictions available")
    else:
        metrics["roc_auc"] = roc_auc_score(y, y_prob)

    # Log metrics
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    if "roc_auc" in metrics:
        logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    # Log to MLflow if active run exists
    if mlflow.active_run():
        mlflow.log_metrics(metrics)

    # Generate confusion matrix
    cm = confusion_matrix(y, y_pred)

    if output_dir:
        # Plot and save confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Purchase", "Purchase"],
            yticklabels=["No Purchase", "Purchase"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()

        # Log artifact to MLflow if active run exists
        if mlflow.active_run():
            mlflow.log_artifact(cm_path)

        # Generate and save ROC curve
        if not np.array_equal(y_prob, y_pred):
            fpr, tpr, _ = roc_curve(y, y_prob)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.4f})')
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.grid(True, alpha=0.3)
            roc_path = os.path.join(output_dir, "roc_curve.png")
            plt.savefig(roc_path)
            plt.close()

            # Log artifact to MLflow if active run exists
            if mlflow.active_run():
                mlflow.log_artifact(roc_path)

            # Generate and save Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y, y_prob)
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve")
            plt.grid(True, alpha=0.3)
            pr_path = os.path.join(output_dir, "precision_recall_curve.png")
            plt.savefig(pr_path)
            plt.close()

            # Log artifact to MLflow if active run exists
            if mlflow.active_run():
                mlflow.log_artifact(pr_path)

    return metrics


def evaluate_threshold(model, X, y, thresholds=None, output_dir=None):
    """
    Evaluate model performance at different probability thresholds

    Args:
        model: Trained model
        X: Feature matrix
        y: True labels
        thresholds (list): List of thresholds to evaluate
        output_dir (str): Directory to save outputs

    Returns:
        pd.DataFrame: Threshold evaluation results
    """
    logger.info("Evaluating model at different probability thresholds")

    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)

    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Make predictions
    try:
        y_prob = model.predict_proba(X)[:, 1]
    except:
        logger.warning(
            "Model doesn't support predict_proba, skipping threshold evaluation"
        )
        return None

    # Evaluate at each threshold
    results = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        # Calculate metrics
        metrics = {
            "threshold": threshold,
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
        }

        results.append(metrics)

    # Create dataframe
    df_results = pd.DataFrame(results)

    if output_dir:
        # Plot and save threshold evaluation
        plt.figure(figsize=(10, 6))
        for col in ["accuracy", "precision", "recall", "f1"]:
            plt.plot(df_results["threshold"], df_results[col], marker="o", label=col)
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("Metric Scores at Different Thresholds")
        plt.grid(True, alpha=0.3)
        plt.legend()
        threshold_path = os.path.join(output_dir, "threshold_evaluation.png")
        plt.savefig(threshold_path)
        plt.close()

        # Save results to CSV
        csv_path = os.path.join(output_dir, "threshold_evaluation.csv")
        df_results.to_csv(csv_path, index=False)

        # Log artifacts to MLflow if active run exists
        if mlflow.active_run():
            mlflow.log_artifact(threshold_path)
            mlflow.log_artifact(csv_path)

    return df_results


def generate_classification_report(model, X, y, output_dir=None):
    """
    Generate a detailed classification report

    Args:
        model: Trained model
        X: Feature matrix
        y: True labels
        output_dir (str): Directory to save outputs

    Returns:
        str: Classification report text
    """
    logger.info("Generating classification report")

    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Make predictions
    y_pred = model.predict(X)

    # Generate report
    report = classification_report(y, y_pred)
    logger.info(f"Classification Report:\n{report}")

    if output_dir:
        # Save report to file
        report_path = os.path.join(output_dir, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)

        # Log artifact to MLflow if active run exists
        if mlflow.active_run():
            mlflow.log_artifact(report_path)

    return report
