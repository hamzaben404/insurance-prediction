# src/models/model_factory.py
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def create_model(model_type, params=None):
    """
    Create a model instance based on model type

    Args:
        model_type (str): Type of model to create
        params (dict): Model parameters

    Returns:
        model: Instantiated model
    """
    # Use default parameters if none provided
    if params is None:
        params = {}

    logger.info(f"Creating {model_type} model with params: {params}")

    # Create model based on type
    if model_type == "logistic_regression":
        default_params = {"C": 1.0, "max_iter": 1000, "random_state": 42}
        default_params.update(params)
        return LogisticRegression(**default_params)

    elif model_type == "random_forest":
        default_params = {"n_estimators": 100, "random_state": 42}
        default_params.update(params)
        return RandomForestClassifier(**default_params)

    elif model_type == "gradient_boosting":
        default_params = {"n_estimators": 100, "random_state": 42}
        default_params.update(params)
        return GradientBoostingClassifier(**default_params)

    elif model_type == "knn":
        default_params = {"n_neighbors": 5}
        default_params.update(params)
        return KNeighborsClassifier(**default_params)

    elif model_type == "svm":
        default_params = {"probability": True, "random_state": 42}
        default_params.update(params)
        return SVC(**default_params)

    elif model_type == "xgboost":
        default_params = {"n_estimators": 100, "random_state": 42}
        default_params.update(params)
        return XGBClassifier(**default_params)

    elif model_type == "lightgbm":
        default_params = {"n_estimators": 100, "random_state": 42}
        default_params.update(params)
        return LGBMClassifier(**default_params)

    elif model_type == "catboost":
        default_params = {"iterations": 100, "random_seed": 42, "verbose": False}
        default_params.update(params)
        return CatBoostClassifier(**default_params)

    else:
        logger.error(f"Unsupported model type: {model_type}")
        raise ValueError(f"Unsupported model type: {model_type}")


def get_available_models():
    """
    Get list of available models

    Returns:
        list: List of available model types
    """
    return [
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "knn",
        "svm",
        "xgboost",
        "lightgbm",
        "catboost",
    ]
