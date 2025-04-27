# src/data/validate_data.py
import numpy as np
import pandas as pd

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def validate_dataset(df):
    """
    Validate a dataset with basic checks

    Args:
        df (pd.DataFrame): Dataframe to validate

    Returns:
        tuple: (bool, dict) - validation success and results
    """
    logger.info("Validating dataset")

    # Initialize validation results
    validation_results = {"checks": [], "success": True}

    # Check 1: Required columns exist
    required_columns = [
        "id",
        "Gender",
        "Age",
        "HasDrivingLicense",
        "RegionID",
        "VehicleAge",
        "AnnualPremium",
        "SalesChannelID",
        "DaysSinceCreated",
    ]

    for col in required_columns:
        check_result = {"check": f"Column {col} exists", "success": col in df.columns}
        validation_results["checks"].append(check_result)
        if not check_result["success"]:
            validation_results["success"] = False

    # Check 2: No duplicate IDs
    check_result = {"check": "No duplicate IDs", "success": df["id"].is_unique}
    validation_results["checks"].append(check_result)
    if not check_result["success"]:
        validation_results["success"] = False

    # Check 3: Age is within reasonable range
    if "Age" in df.columns:
        min_age = df["Age"].min()
        max_age = df["Age"].max()
        check_result = {
            "check": "Age is within reasonable range (16-100)",
            "success": (min_age >= 16) and (max_age <= 100),
        }
        validation_results["checks"].append(check_result)
        if not check_result["success"]:
            validation_results["success"] = False

    # Check 4: Gender values are valid
    if "Gender" in df.columns:
        valid_genders = ["Male", "Female"]
        invalid_genders = df["Gender"].dropna().unique().tolist()
        invalid_genders = [g for g in invalid_genders if g not in valid_genders]

        check_result = {
            "check": "Gender values are valid",
            "success": len(invalid_genders) == 0,
        }
        if not check_result["success"]:
            check_result["invalid_values"] = invalid_genders

        validation_results["checks"].append(check_result)
        if not check_result["success"]:
            validation_results["success"] = False

    # Check 5: AnnualPremium is non-negative
    if "AnnualPremium" in df.columns:
        # Since AnnualPremium might have currency symbols, we check after cleaning
        premiums = (
            df["AnnualPremium"].astype(str).str.replace("£", "").str.replace(",", "")
        )
        premiums = pd.to_numeric(premiums, errors="coerce")
        min_premium = premiums.min()

        check_result = {
            "check": "AnnualPremium is non-negative",
            "success": min_premium >= 0,
        }
        validation_results["checks"].append(check_result)
        if not check_result["success"]:
            validation_results["success"] = False

    # Log validation results
    if validation_results["success"]:
        logger.info("Data validation passed all checks")
    else:
        logger.warning("Data validation failed some checks")
        for check in validation_results["checks"]:
            if not check["success"]:
                logger.warning(f"Failed check: {check['check']}")

    return validation_results["success"], validation_results
