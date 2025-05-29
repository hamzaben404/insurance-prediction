# src/data/preprocess.py
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

# def clean_column_names(df):
#     """Standardize column names to snake_case"""
#     # Create a mapping of original column names to standardized column names
#     column_mapping = {
#         'id': 'id',
#         'Gender': 'gender',
#         'Age': 'age',
#         'HasDrivingLicense': 'has_driving_license',
#         'RegionID': 'region_id',
#         'Switch': 'switch',
#         'VehicleAge': 'vehicle_age',
#         'PastAccident': 'past_accident',
#         'AnnualPremium': 'annual_premium',
#         'SalesChannelID': 'sales_channel_id',
#         'DaysSinceCreated': 'days_since_created',
#         'Result': 'result'
#     }

#     # Rename columns that exist in the dataframe
#     cols_to_rename = {col: column_mapping[col] for col in df.columns if col in column_mapping}
#     df = df.rename(columns=cols_to_rename)

#     logger.info(f"Standardized column names: {df.columns.tolist()}")
#     return df

# def clean_column_names(df):
# """Standardize column names to snake_case"""
# # Create a mapping for non-ASCII characters
# import re

# # Function to convert a column name to snake_case and handle special characters
# def normalize_column_name(col):
#     # Replace special characters with underscore
#     col = re.sub(r'[^a-zA-Z0-9]', '_', col)
#     # Convert to lowercase
#     col = col.lower()
#     # Replace multiple underscores with a single one
#     col = re.sub(r'_+', '_', col)
#     # Remove leading/trailing underscores
#     col = col.strip('_')
#     return col

# # Process each column name
# new_columns = [normalize_column_name(col) for col in df.columns]

# # Update DataFrame columns
# df.columns = new_columns

# logger.info(f"Standardized column names: {df.columns.tolist()}")
# return df


def clean_column_names(df):
    """Standardize column names to snake_case"""
    import re

    def normalize_column_name(col):
        # Convert camelCase to snake_case
        col = re.sub(r"(?<!^)(?=[A-Z])", "_", col)
        # Replace special characters with underscores
        col = re.sub(r"[^a-zA-Z0-9_]", "_", col)
        # Convert to lowercase and clean underscores
        col = col.lower().strip("_")
        col = re.sub(r"_+", "_", col)
        return col

    df.columns = [normalize_column_name(col) for col in df.columns]
    logger.info(f"Standardized column names: {df.columns.tolist()}")
    return df


# def clean_currency_values(df, column='annual_premium'):
#     """Clean currency values by removing symbols and commas"""
#     if column in df.columns:
#         logger.info(f"Cleaning currency values in {column}")
#         # Handle the specific format with pound symbol and commas
#         df[column] = df[column].astype(str).str.replace('£', '', regex=False)
#         df[column] = df[column].str.replace(',', '', regex=False)
#         df[column] = df[column].str.strip()
#         df[column] = pd.to_numeric(df[column], errors='coerce')
#         logger.info(f"Converted {column} to numeric format")
#     return df

# def clean_currency_values(df, column='annual_premium'):
#     """Clean currency values by removing symbols and commas"""
#     if column in df.columns:
#         logger.info(f"Cleaning currency values in {column}")
#         # Handle the specific format with pound symbol and commas
#         df[column] = df[column].astype(str).str.replace('£', '', regex=False)
#         df[column] = df[column].str.replace(',', '', regex=False)
#         df[column] = df[column].str.strip()
#         # Force float conversion
#         df[column] = pd.to_numeric(df[column], errors='coerce').astype(float)
#         logger.info(f"Converted {column} to numeric format")
#     return df


# src/data/preprocess.py
def clean_currency_values(df, column="annual_premium"):
    """Clean currency values by removing symbols and commas"""
    if column in df.columns:
        logger.info(f"Cleaning currency values in {column}")
        # Handle multiple currency formats
        df[column] = (
            df[column]
            .astype(str)
            .str.replace(r"[£,€$]", "", regex=True)
            .str.replace(r"\s+", "", regex=True)
            .replace("", "0")
            .astype(float)
        )
        logger.info(f"Converted {column} to numeric format")
    return df


def handle_missing_values(df):
    """Handle missing values in the dataframe"""
    # Log missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        logger.info(f"Missing values before imputation:\n{missing}")

    # Handle missing values for each column appropriately
    # Numerical columns - fill with median
    for col in ["age", "annual_premium", "days_since_created"]:
        if col in df.columns and df[col].isnull().any():
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            logger.info(f"Filled missing values in {col} with median: {median_value}")

    # Categorical columns - fill with mode
    for col in ["gender", "vehicle_age", "past_accident"]:
        if col in df.columns and df[col].isnull().any():
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)
            logger.info(f"Filled missing values in {col} with mode: {mode_value}")

    # Binary columns - fill with 0 (assuming 0 is the default state)
    for col in ["has_driving_license", "switch"]:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(0)
            logger.info(f"Filled missing values in {col} with 0")

    return df


def preprocess_data(df):
    """Run full preprocessing pipeline on dataframe"""
    logger.info("Starting data preprocessing")

    # Make a copy to avoid modifying the original
    df = df.copy()

    # Clean column names
    df = clean_column_names(df)

    # Clean currency values
    df = clean_currency_values(df)

    # Handle missing values
    df = handle_missing_values(df)

    logger.info("Preprocessing completed")
    return df
