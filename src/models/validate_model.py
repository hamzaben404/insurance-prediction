# src/models/validate_model.py
import argparse
import os
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


def main():
    """Validate model metrics against thresholds"""
    parser = argparse.ArgumentParser(description="Validate model metrics")
    parser.add_argument("--data", required=True, help="Path to test data CSV")
    parser.add_argument("--threshold", type=float, default=0.7, help="Minimum acceptable metric")

    args = parser.parse_args()

    # Load test data
    df = pd.read_csv(args.data)

    # Assume the last column is the target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Load model
    model_path = os.environ.get("MODEL_PATH", "models/comparison/production/production_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Make predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    try:
        auc = roc_auc_score(y, y_prob)
    except (ValueError, TypeError):  # Specify the exception types
        auc = 0

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {auc:.4f}")

    # Check against threshold
    if accuracy < args.threshold and auc < args.threshold:
        print(f"Model performance below threshold {args.threshold}")
        exit(1)

    print("Model validation passed!")


if __name__ == "__main__":
    main()
