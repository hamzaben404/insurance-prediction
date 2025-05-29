# tests/end_to_end_test.py
import os
import sys

from src.data.data_split import split_data
from src.data.load_data import load_dataset
from src.data.preprocess import preprocess_data
from src.data.validate_data import validate_dataset
from src.features.build_features import create_feature_pipeline

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Test paths
train_path = "data/raw/train.csv"
output_path = "data/processed/test_output"

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Step 1: Load data
print("Loading data...")
df = load_dataset(train_path)
print(f"Loaded data with shape: {df.shape}")
print(f"Sample data:\n{df.head()}")
print("\n" + "-" * 50 + "\n")

# Step 2: Validate data
print("Validating data...")
valid, results = validate_dataset(df)
print(f"Data validation result: {'Passed' if valid else 'Failed'}")
print("\n" + "-" * 50 + "\n")

# Step 3: Preprocess data
print("Preprocessing data...")
df_processed = preprocess_data(df)
print(f"Processed data shape: {df_processed.shape}")
print(f"Processed data columns: {df_processed.columns.tolist()}")
print(f"Processed data sample:\n{df_processed.head()}")
print("\n" + "-" * 50 + "\n")

# Step 4: Feature engineering
print("Applying feature engineering...")
df_featured = create_feature_pipeline(df_processed)
print(f"Featured data shape: {df_featured.shape}")
print(f"Featured data columns: {df_featured.columns.tolist()}")
print(f"Featured data sample:\n{df_featured.head()}")
print("\n" + "-" * 50 + "\n")

# Step 5: Split data
print("Splitting data...")
try:
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_featured)
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation shapes: X={X_val.shape}, y={y_val.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
except Exception as e:
    print(f"Error splitting data: {e}")

# Step 6: Save outputs
df_featured.to_csv(f"{output_path}/featured_data.csv", index=False)
print(f"\nTest complete! Featured data saved to {output_path}/featured_data.csv")
