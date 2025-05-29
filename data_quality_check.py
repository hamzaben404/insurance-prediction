# data_quality_check.py
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load processed data
processed_data = pd.read_csv("data/processed/test_output/featured_data.csv")

# Create output directory for visualizations
os.makedirs("reports/data_quality", exist_ok=True)

# 1. Check for missing values
missing = processed_data.isnull().sum()
print("Missing values in processed data:")
print(missing[missing > 0])

# 2. Check for distribution of numerical features
numerical_cols = processed_data.select_dtypes(include=["float64", "int64"]).columns
for col in numerical_cols[:5]:  # First 5 columns only for brevity
    plt.figure(figsize=(8, 4))
    sns.histplot(processed_data[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.savefig(f"reports/data_quality/{col}_distribution.png")
    plt.close()

# 3. Check correlation matrix
plt.figure(figsize=(12, 10))
corr = processed_data.select_dtypes(include=["float64", "int64"]).corr()
sns.heatmap(corr, annot=False, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig("reports/data_quality/correlation_matrix.png")
plt.close()

print("Data quality checks complete. Visualizations saved to reports/data_quality/")
