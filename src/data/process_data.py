# src/data/process_data.py
import argparse
import json
from pathlib import Path

from config.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.data_split import split_data
from src.data.load_data import get_data_info, load_dataset
from src.data.preprocess import preprocess_data
from src.data.validate_data import validate_dataset
from src.features.build_features import create_feature_pipeline
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def process_pipeline(input_file, output_dir=None, create_splits=True):
    """
    Run the complete data processing pipeline

    Args:
        input_file (str): Path to input CSV file
        output_dir (str, optional): Directory to save processed files
        create_splits (bool): Whether to create train/val/test splits

    Returns:
        dict: Processing results and metadata
    """
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    logger.info(f"Loading data from {input_file}")
    df = load_dataset(input_file)

    # Step 2: Get data info and save it
    data_info = get_data_info(df)
    with open(f"{output_dir}/data_info.json", "w") as f:
        json.dump(data_info, f, indent=4, default=str)

    # Step 3: Validate raw data
    logger.info("Validating raw data")
    raw_validation_success, _ = validate_dataset(df)

    # Step 4: Preprocess data
    logger.info("Preprocessing data")
    df_processed = preprocess_data(df)

    # Step 5: Create data profile
    logger.info("Creating data profile")
    # profile = create_data_profile(df_processed, output_dir=f"{output_dir}/profile")

    # Step 6: Feature engineering
    logger.info("Running feature engineering")
    df_featured = create_feature_pipeline(df_processed)

    # Save processed data
    processed_file = f"{output_dir}/processed_data.csv"
    df_featured.to_csv(processed_file, index=False)
    logger.info(f"Saved processed data to {processed_file}")

    # Step 7: Create data splits if requested
    if create_splits and "result" in df_featured.columns:
        logger.info("Creating train/validation/test splits")
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_featured)

        # Save splits
        X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
        X_val.to_csv(f"{output_dir}/X_val.csv", index=False)
        X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
        y_val.to_csv(f"{output_dir}/y_val.csv", index=False)
        y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

        logger.info(f"Saved data splits to {output_dir}")

    # Return results
    results = {
        "input_file": input_file,
        "output_dir": output_dir,
        "raw_shape": df.shape,
        "processed_shape": df_featured.shape,
        "validation_success": raw_validation_success,
        "created_splits": create_splits,
    }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data processing pipeline")
    parser.add_argument(
        "--input",
        type=str,
        default=f"{RAW_DATA_DIR}/train.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=PROCESSED_DATA_DIR,
        help="Directory to save processed files",
    )
    parser.add_argument(
        "--no-splits",
        action="store_false",
        dest="create_splits",
        help="Skip creating train/val/test splits",
    )

    args = parser.parse_args()

    # Run pipeline
    process_pipeline(args.input, args.output, args.create_splits)
