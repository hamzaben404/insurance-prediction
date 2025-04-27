# src/models/interpretation.py
import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.inspection import permutation_importance

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def plot_feature_importance(model, X, feature_names=None, top_n=20, output_dir=None):
    """
    Plot feature importance for a trained model

    Args:
        model: Trained model
        X: Feature matrix
        feature_names (list): List of feature names
        top_n (int): Number of top features to show
        output_dir (str): Directory to save outputs

    Returns:
        pd.DataFrame: Feature importance dataframe
    """
    logger.info("Plotting feature importance")

    # Use feature names from X if not provided
    if feature_names is None:
        if hasattr(X, "columns"):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Extract feature importance based on model type
    importances = None

    # For models with feature_importances_ attribute
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    # For linear models with coef_ attribute
    elif hasattr(model, "coef_"):
        importances = (
            np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
        )

    # Use permutation importance if importances not available
    if importances is None:
        logger.info(
            "Model doesn't have built-in feature importance, using permutation importance"
        )
        perm_importance = permutation_importance(
            model, X, method="predict", n_repeats=10, random_state=42
        )
        importances = perm_importance.importances_mean

    # Create and sort importance dataframe
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    importance_df = importance_df.sort_values(
        "Importance", ascending=False
    ).reset_index(drop=True)

    # Limit to top N features
    if len(importance_df) > top_n:
        importance_df = importance_df.head(top_n)

    if output_dir:
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        sns.barplot(x="Importance", y="Feature", data=importance_df)
        plt.title(f"Top {top_n} Feature Importance")
        plt.tight_layout()
        importance_path = os.path.join(output_dir, "feature_importance.png")
        plt.savefig(importance_path)
        plt.close()

        # Save importance to CSV
        csv_path = os.path.join(output_dir, "feature_importance.csv")
        importance_df.to_csv(csv_path, index=False)

        # Log artifacts to MLflow if active run exists
        if mlflow.active_run():
            mlflow.log_artifact(importance_path)
            mlflow.log_artifact(csv_path)

    return importance_df


def generate_shap_explanation(model, X, X_background=None, output_dir=None):
    """
    Generate SHAP values and plots for model explanation

    Args:
        model: Trained model
        X: Feature matrix for explanation
        X_background: Background dataset for SHAP (if None, uses a sample of X)
        output_dir (str): Directory to save outputs

    Returns:
        shap.Explanation: SHAP explanation object
    """
    logger.info("Generating SHAP explanations")

    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Convert data to numpy if it's pandas
    X_data = X.copy()
    feature_names = X.columns.tolist() if hasattr(X, "columns") else None

    # Sample data for background if not provided
    if X_background is None:
        # Use a sample of X if X is large
        if len(X) > 100:
            X_background = X.sample(100, random_state=42)
        else:
            X_background = X

    try:
        # Create explainer based on model type
        if hasattr(model, "predict_proba"):
            # Tree models can use TreeExplainer
            if hasattr(model, "estimators_") or type(model).__name__ in [
                "XGBClassifier",
                "LGBMClassifier",
                "CatBoostClassifier",
            ]:
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.KernelExplainer(
                    lambda x: model.predict_proba(x)[:, 1], X_background
                )
        else:
            explainer = shap.KernelExplainer(model.predict, X_background)

        # Calculate SHAP values
        shap_values = explainer.shap_values(X_data)

        # For models that return a list of shap values (e.g., for each class)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification, use class 1

        if output_dir:
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values, X_data, feature_names=feature_names, show=False
            )
            summary_path = os.path.join(output_dir, "shap_summary.png")
            plt.savefig(summary_path)
            plt.close()

            # Bar plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values,
                X_data,
                feature_names=feature_names,
                plot_type="bar",
                show=False,
            )
            bar_path = os.path.join(output_dir, "shap_bar.png")
            plt.savefig(bar_path)
            plt.close()

            # Sample dependence plots for top 3 features
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top_indices = np.argsort(mean_abs_shap)[-3:]

            for i, idx in enumerate(top_indices):
                feature_name = feature_names[idx] if feature_names else f"feature_{idx}"
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(
                    idx, shap_values, X_data, feature_names=feature_names, show=False
                )
                plt.title(f"SHAP Dependence Plot for {feature_name}")
                dep_path = os.path.join(output_dir, f"shap_dependence_{i}.png")
                plt.savefig(dep_path)
                plt.close()

            # Log artifacts to MLflow if active run exists
            if mlflow.active_run():
                mlflow.log_artifact(summary_path)
                mlflow.log_artifact(bar_path)
                for i in range(len(top_indices)):
                    mlflow.log_artifact(
                        os.path.join(output_dir, f"shap_dependence_{i}.png")
                    )

        return {"shap_values": shap_values, "explainer": explainer}

    except Exception as e:
        logger.error(f"Error generating SHAP explanation: {e}")
        return None


def plot_partial_dependence(model, X, features=None, output_dir=None):
    """
    Generate partial dependence plots for selected features

    Args:
        model: Trained model
        X: Feature matrix
        features (list): List of feature indices or names
        output_dir (str): Directory to save outputs

    Returns:
        None
    """
    from sklearn.inspection import partial_dependence, plot_partial_dependence

    logger.info("Generating partial dependence plots")

    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Get feature names
    if hasattr(X, "columns"):
        feature_names = X.columns.tolist()
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # If features not specified, use top 5 important features
    if features is None:
        # Get feature importance
        importance_df = plot_feature_importance(
            model, X, feature_names, output_dir=None
        )
        features = importance_df["Feature"].head(5).tolist()

    # Convert feature names to indices if needed
    if isinstance(features[0], str):
        feature_indices = [feature_names.index(f) for f in features]
    else:
        feature_indices = features
        features = [feature_names[i] for i in feature_indices]

    try:
        # Generate partial dependence plot for each feature
        for i, (feature_idx, feature_name) in enumerate(zip(feature_indices, features)):
            # Calculate partial dependence
            pdp_result = partial_dependence(model, X, [feature_idx])

            # Plot
            plt.figure(figsize=(8, 6))

            # Get values and partial dependence
            values = pdp_result["values"][0]
            pdp = pdp_result["average"][0]

            plt.plot(values, pdp)
            plt.xlabel(feature_name)
            plt.ylabel("Partial Dependence")
            plt.title(f"Partial Dependence Plot for {feature_name}")
            plt.grid(True, alpha=0.3)

            # Save plot
            pdp_path = os.path.join(output_dir, f"pdp_{feature_name}.png")
            plt.savefig(pdp_path)
            plt.close()

            # Log artifact to MLflow if active run exists
            if mlflow.active_run():
                mlflow.log_artifact(pdp_path)

    except Exception as e:
        logger.error(f"Error generating partial dependence plots: {e}")
