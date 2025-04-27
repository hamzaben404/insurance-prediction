# fill_feature_documentation.py
import pandas as pd
import json
import os
from pathlib import Path

# Create docs directory if it doesn't exist
os.makedirs('docs/features', exist_ok=True)

# Load processed data
processed_df = pd.read_csv('data/processed/test_output/featured_data.csv')

# Create feature documentation template
template = """# Feature Documentation

## Dataset Overview
- **Source**: Vehicle Insurance Prediction
- **Target Variable**: `result` (binary: 0 or 1)
- **Number of Records**: {num_records}
- **Number of Features**: {num_features}

## Feature Descriptions

### Original Features

| Feature Name | Description | Data Type | Range/Values | Missing Values |
|--------------|-------------|-----------|------------|---------------|
{original_features}

### Engineered Features

| Feature Name | Description | Derivation | Data Type |
|--------------|-------------|------------|-----------|
{engineered_features}

## Feature Preprocessing

1. **Cleaning**:
   - Standardized column names to snake_case
   - Removed currency symbols and commas from monetary values
   - Handled missing values using median/mode imputation

2. **Encoding**:
   - Categorical features encoded using one-hot encoding
   - Binary features converted to 0/1 numerical values

3. **Scaling**:
   - Numerical features normalized using StandardScaler

## Feature Statistics

{feature_stats}

## Notes and Recommendations

- Consider additional feature engineering based on domain knowledge
- Further analysis of feature importance recommended after model training
"""

# Original features
original_cols = [
    'id', 'gender', 'age', 'has_driving_license', 'region_id', 
    'switch', 'vehicle_age', 'past_accident', 'annual_premium', 
    'sales_channel_id', 'days_since_created', 'result'
]

# Get only columns that exist in processed_df
original_cols = [col for col in original_cols if col in processed_df.columns]

original_features = ""
for col in original_cols:
    if col in processed_df.columns:
        dtype = str(processed_df[col].dtype)
        missing = processed_df[col].isnull().sum()
        missing_pct = round(missing / len(processed_df) * 100, 2)
        
        if pd.api.types.is_numeric_dtype(processed_df[col]):
            value_range = f"{processed_df[col].min()} to {processed_df[col].max()}"
        else:
            unique_vals = processed_df[col].unique()
            if len(unique_vals) <= 5:
                value_range = ", ".join(str(x) for x in unique_vals)
            else:
                value_range = f"{len(unique_vals)} unique values"
                
        original_features += f"| {col} | | {dtype} | {value_range} | {missing_pct}% |\n"

# Engineered features (any column not in original_cols)
engineered_cols = [col for col in processed_df.columns if col not in original_cols]
engineered_features = ""
for col in engineered_cols:
    dtype = str(processed_df[col].dtype)
    
    # Try to determine derivation based on column name
    derivation = ""
    if "ratio" in col:
        derivation = "Ratio calculation"
    elif col.startswith("age_"):
        derivation = "Age-based calculation"
    elif "_encoded" in col:
        derivation = "One-hot encoding"
    
    engineered_features += f"| {col} | | {derivation} | {dtype} |\n"

# Feature statistics
feature_stats = "### Numerical Feature Statistics\n\n"
feature_stats += "| Feature | Mean | Median | Std Dev | Min | Max |\n"
feature_stats += "|---------|------|--------|---------|-----|-----|\n"

numerical_cols = processed_df.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_cols[:10]:  # Limit to first 10 for brevity
    stats = processed_df[col].describe()
    feature_stats += f"| {col} | {stats['mean']:.2f} | {stats['50%']:.2f} | {stats['std']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} |\n"

# Fill in the template
filled_template = template.format(
    num_records=len(processed_df),
    num_features=len(processed_df.columns),
    original_features=original_features,
    engineered_features=engineered_features,
    feature_stats=feature_stats
)

# Write the filled template
with open('docs/features/feature_documentation.md', 'w') as f:
    f.write(filled_template)

print("Feature documentation created at docs/features/feature_documentation.md")