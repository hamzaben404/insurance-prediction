# run_phase3_pipeline.py
import os
import pandas as pd
import numpy as np
import sys
import argparse

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.utils.logging import setup_logger
from src.models.mlflow_utils import setup_mlflow
from src.models.model_selection import train_multiple_models, select_best_model

# Set up logger
logger = setup_logger("phase3_pipeline")

def main(args):
    """Run the complete Phase 3 pipeline with processed data from Phase 2"""
    
    logger.info("Starting Phase 3 pipeline - Model Development & Experiment Tracking")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup MLflow
    experiment_id = setup_mlflow(args.experiment_name)
    logger.info(f"MLflow experiment set up with ID: {experiment_id}")
    
    # Load processed data from Phase 2
    logger.info(f"Loading processed data from {args.data_dir}")
    try:
        X_train = pd.read_csv(os.path.join(args.data_dir, "X_train.csv"))
        y_train = pd.read_csv(os.path.join(args.data_dir, "y_train.csv"))
        X_val = pd.read_csv(os.path.join(args.data_dir, "X_val.csv"))
        y_val = pd.read_csv(os.path.join(args.data_dir, "y_val.csv"))
        
        # Check if y is a dataframe and extract the series
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.iloc[:, 0]
        if isinstance(y_val, pd.DataFrame):
            y_val = y_val.iloc[:, 0]
            
        logger.info(f"Data loaded successfully - Shapes: X_train={X_train.shape}, y_train={y_train.shape}")
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        logger.error("Make sure you've completed Phase 2 and have processed data files")
        sys.exit(1)
    
    """
    # Define models to train
    if args.quick_test:
        # Quick test with just 2 simple models
        # models_to_try = ["logistic_regression", "random_forest"]
    else:
        # Full model suite
        models_to_try = [
            "logistic_regression", 
            "random_forest", 
            "gradient_boosting", 
            "xgboost",
            "lightgbm"
        ]
    """
    
    models_to_try = [
        "logistic_regression", 
        "random_forest", 
        "gradient_boosting", 
        "xgboost",
        "lightgbm"
    ]
    
    logger.info(f"Training models: {models_to_try}")
    
    # Train and compare models
    comparison_df = train_multiple_models(
        X_train, y_train, X_val, y_val, 
        models_to_try=models_to_try,
        base_output_dir=args.output_dir
    )
    
    # Select best model
    best_model_type, best_model_path = select_best_model(
        comparison_df, 
        metric=args.metric
    )
    
    # Create a final production model directory
    prod_dir = os.path.join(args.output_dir, "production")
    os.makedirs(prod_dir, exist_ok=True)
    
    # Copy the best model to production folder
    import shutil
    prod_model_path = os.path.join(prod_dir, "production_model.pkl")
    shutil.copy(best_model_path, prod_model_path)
    
    # Save model info
    with open(os.path.join(prod_dir, "model_info.txt"), "w") as f:
        f.write(f"Best model type: {best_model_type}\n")
        f.write(f"Original model path: {best_model_path}\n")
        f.write(f"Selected based on {args.metric}\n")
    
    logger.info("="*50)
    logger.info(f"Phase 3 completed successfully!")
    logger.info(f"Best model: {best_model_type}")
    logger.info(f"Production model saved at: {prod_model_path}")
    logger.info(f"Full results in: {args.output_dir}")
    logger.info("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Phase 3 model training pipeline")
    
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data/processed/train",
        help="Directory containing processed data from Phase 2"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="models/comparison",
        help="Directory to save models and results"
    )
    
    parser.add_argument(
        "--experiment-name", 
        type=str, 
        default="insurance_prediction",
        help="MLflow experiment name"
    )
    
    parser.add_argument(
        "--metric", 
        type=str, 
        default="val_f1",
        help="Metric to use for model selection"
    )
    
    parser.add_argument(
        "--quick-test", 
        action="store_true",
        help="Run a quick test with fewer models"
    )
    
    args = parser.parse_args()
    
    main(args)