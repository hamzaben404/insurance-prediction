# src/models/hyperparameter_tuning.py
import mlflow
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def grid_search_cv(model, param_grid, X, y, cv=5, scoring="roc_auc", refit=True, random_state=42):
    """
    Perform grid search cross-validation

    Args:
        model: Base model
        param_grid (dict): Parameter grid
        X: Feature matrix
        y: Target vector
        cv (int): Number of folds
        scoring (str): Scoring metric
        refit (bool): Whether to refit with best params
        random_state (int): Random state for reproducibility

    Returns:
        tuple: (best_model, best_params, best_score)
    """
    logger.info(f"Performing grid search with {len(param_grid)} parameters")

    # Create grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit=refit,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    # Fit grid search
    grid_search.fit(X, y)

    # Get best results
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best {scoring} score: {best_score:.4f}")

    # Log to MLflow if active run exists
    if mlflow.active_run():
        mlflow.log_params(best_params)
        mlflow.log_metric(f"best_{scoring}", best_score)

        # Log all results as a dataframe
        cv_results = grid_search.cv_results_
        df_results = pd.DataFrame(cv_results)
        mlflow.log_artifact(df_results.to_csv("grid_search_results.csv"))

    # Return best model
    if refit:
        return grid_search.best_estimator_, best_params, best_score
    else:
        return None, best_params, best_score


def random_search_cv(
    model,
    param_distributions,
    X,
    y,
    cv=5,
    scoring="roc_auc",
    n_iter=10,
    refit=True,
    random_state=42,
):
    """
    Perform random search cross-validation

    Args:
        model: Base model
        param_distributions (dict): Parameter distributions
        X: Feature matrix
        y: Target vector
        cv (int): Number of folds
        scoring (str): Scoring metric
        n_iter (int): Number of iterations
        refit (bool): Whether to refit with best params
        random_state (int): Random state for reproducibility

    Returns:
        tuple: (best_model, best_params, best_score)
    """
    logger.info(f"Performing random search with {n_iter} iterations")

    # Create random search
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        refit=refit,
        n_jobs=-1,
        verbose=1,
        random_state=random_state,
        return_train_score=True,
    )

    # Fit random search
    random_search.fit(X, y)

    # Get best results
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best {scoring} score: {best_score:.4f}")

    # Log to MLflow if active run exists
    if mlflow.active_run():
        mlflow.log_params(best_params)
        mlflow.log_metric(f"best_{scoring}", best_score)

        # Log all results as a dataframe
        cv_results = random_search.cv_results_
        import pandas as pd

        df_results = pd.DataFrame(cv_results)
        mlflow.log_artifact(df_results.to_csv("random_search_results.csv"))

    # Return best model
    if refit:
        return random_search.best_estimator_, best_params, best_score
    else:
        return None, best_params, best_score


def get_default_param_grids():
    """
    Get default parameter grids for different models

    Returns:
        dict: Default parameter grids
    """
    param_grids = {
        "logistic_regression": {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"],
        },
        "random_forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "gradient_boosting": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        },
        "xgboost": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.7, 0.8, 0.9],
        },
        "lightgbm": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "num_leaves": [31, 50, 70],
        },
        "catboost": {
            "iterations": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "depth": [4, 6, 8],
        },
    }

    return param_grids
