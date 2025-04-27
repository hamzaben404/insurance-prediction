# src/utils/mlflow_utils.py
import os
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def setup_mlflow(experiment_name="insurance_prediction", tracking_uri=None):
    """
    Set up MLflow tracking

    Args:
        experiment_name (str): Name of the MLflow experiment
        tracking_uri (str): MLflow tracking URI (local or remote)

    Returns:
        str: Active experiment ID
    """
    # Set tracking URI if provided
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # Get or create experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
            logger.info(
                f"Using existing experiment '{experiment_name}' (ID: {experiment_id})"
            )
        else:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=os.path.join("mlruns", experiment_name),
            )
            logger.info(
                f"Created new experiment '{experiment_name}' (ID: {experiment_id})"
            )

        return experiment_id
    except Exception as e:
        logger.error(f"Error setting up MLflow: {e}")
        # If MLflow setup fails, return None but don't crash
        return None


def start_run(run_name=None):
    """
    Start a new MLflow run

    Args:
        run_name (str): Name for the run

    Returns:
        mlflow.ActiveRun: Active MLflow run
    """
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return mlflow.start_run(run_name=run_name)


def log_dataset_info(dataset_info):
    """
    Log dataset information as tags

    Args:
        dataset_info (dict): Dataset information to log
    """
    # Log basic dataset info as tags
    for key, value in dataset_info.items():
        if isinstance(value, (str, int, float, bool)):
            mlflow.set_tag(f"data.{key}", value)


def log_model_params(params):
    """
    Log model parameters

    Args:
        params (dict): Model parameters to log
    """
    for key, value in params.items():
        if isinstance(value, (str, int, float, bool)):
            mlflow.log_param(key, value)


def log_metrics(metrics):
    """
    Log performance metrics

    Args:
        metrics (dict): Metrics to log
    """
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, value)


def register_model(model, name, stage=None):
    """
    Register a model to the MLflow model registry

    Args:
        model: Trained model
        name (str): Model name in the registry
        stage (str): Stage ('None', 'Staging', 'Production', 'Archived')

    Returns:
        str: Model version
    """
    try:
        result = mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model", name
        )
        version = result.version

        # Set stage if provided
        if stage:
            client = MlflowClient()
            client.transition_model_version_stage(
                name=name, version=version, stage=stage
            )
            logger.info(f"Model {name} version {version} transitioned to {stage}")

        return version
    except Exception as e:
        logger.error(f"Error registering model: {e}")
        return None
