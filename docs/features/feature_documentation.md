# Feature Documentation

## Dataset Overview
- **Source**: Vehicle Insurance Prediction
- **Target Variable**: `result` (binary: 0 or 1)
- **Number of Records**: 100000
- **Number of Features**: 18

## Feature Descriptions

### Original Features

| Feature Name | Description | Data Type | Range/Values | Missing Values |
|--------------|-------------|-----------|------------|---------------|
| id | | int64 | 9 to 381103 | 0.0% |
| age | | float64 | -1.2598669481890106 to 3.10047487966511 | 0.0% |
| has_driving_license | | float64 | 0.0 to 1.0 | 0.0% |
| region_id | | float64 | 0.0 to 52.0 | 9.97% |
| switch | | float64 | 0.0 to 1.0 | 0.0% |
| annual_premium | | float64 | -1.6273177890323671 to 29.698189696720195 | 0.0% |
| sales_channel_id | | int64 | 1 to 163 | 0.0% |
| days_since_created | | float64 | -1.7256290184219254 to 1.7307149867667226 | 0.0% |
| result | | int64 | 0 to 1 | 0.0% |


### Engineered Features

| Feature Name | Description | Derivation | Data Type |
|--------------|-------------|------------|-----------|
| gender_Female | |  | bool |
| gender_Male | |  | bool |
| vehicle_age_1to2_Year | |  | bool |
| vehicle_age_less_than_1_Year | |  | bool |
| vehicle_age_more_than_2_Years | |  | bool |
| past_accident_No | |  | bool |
| past_accident_Yes | |  | bool |
| age_premium_ratio | | Ratio calculation | float64 |
| age_days_ratio | | Ratio calculation | float64 |


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

### Numerical Feature Statistics

| Feature | Mean | Median | Std Dev | Min | Max |
|---------|------|--------|---------|-----|-----|
| id | 190907.66 | 190781.50 | 109887.21 | 9.00 | 381103.00 |
| age | -0.00 | -0.17 | 1.00 | -1.26 | 3.10 |
| has_driving_license | 0.90 | 1.00 | 0.30 | 0.00 | 1.00 |
| region_id | 26.38 | 28.00 | 13.22 | 0.00 | 52.00 |
| switch | 0.23 | 0.00 | 0.42 | 0.00 | 1.00 |
| annual_premium | 0.00 | 0.06 | 1.00 | -1.63 | 29.70 |
| sales_channel_id | 112.24 | 139.00 | 54.10 | 1.00 | 163.00 |
| days_since_created | -0.00 | -0.00 | 1.00 | -1.73 | 1.73 |
| result | 0.12 | 0.00 | 0.33 | 0.00 | 1.00 |
| age_premium_ratio | 0.93 | 0.15 | 767.08 | -207491.52 | 78086.89 |


## Notes and Recommendations

- Consider additional feature engineering based on domain knowledge
- Further analysis of feature importance recommended after model training
