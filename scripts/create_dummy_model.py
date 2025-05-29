#!/usr/bin/env python3
"""Create a dummy model for testing purposes"""
import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def create_dummy_model(
    n_features=14, output_path="models/comparison/production/production_model.pkl"
):
    """Create and save a dummy RandomForest model"""
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create dummy data with correct number of features
    X = np.random.rand(100, n_features)
    y = np.random.randint(0, 2, 100)

    # Train model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    # Save model
    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Dummy model created with {n_features} features at {output_path}")


if __name__ == "__main__":
    create_dummy_model()
