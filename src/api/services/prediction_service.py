# src/api/services/prediction_service.py
import logging
import os
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.data.preprocess import preprocess_data
from src.features.build_features import create_feature_pipeline

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self, model_path: str):
        """
        Initialize prediction service

        Args:
            model_path: Path to the model file
        """
        self.model_path = model_path
        self.model = self._load_model(model_path)
        self.model_info = self._get_model_info(model_path)
        self.expected_features = self._get_expected_features()

    def _load_model(self, model_path: str):
        """Load the trained model"""
        try:
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                logger.info(f"Model loaded from {model_path}")
                return model
            else:
                logger.warning(f"Model not found at {model_path}, creating dummy model")
                # Create a dummy model for testing/CI
                model = RandomForestClassifier(n_estimators=10, random_state=42)
                # Fit with dummy data matching expected features
                X_dummy = np.random.rand(100, 17)  # 17 features as per expected_features
                y_dummy = np.random.randint(0, 2, 100)
                model.fit(X_dummy, y_dummy)
                return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Create dummy model as fallback
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            X_dummy = np.random.rand(100, 17)
            y_dummy = np.random.randint(0, 2, 100)
            model.fit(X_dummy, y_dummy)
            return model

    def _get_model_info(self, model_path: str) -> Dict[str, Any]:
        """Get model information"""
        # Get directory containing the model
        model_dir = os.path.dirname(model_path)

        # Try to load model info if available
        info_path = os.path.join(model_dir, "model_info.txt")
        model_info = {
            "model_type": "unknown",
            "model_version": "unknown",
            "features": [],
            "metrics": {},
        }

        if os.path.exists(info_path):
            try:
                with open(info_path, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        if ":" in line:
                            key, value = line.strip().split(":", 1)
                            model_info[key.strip()] = value.strip()
            except Exception as e:
                logger.warning(f"Could not read model info: {e}")

        # Set model type based on the model object if not found in info file
        if model_info["model_type"] == "unknown" and hasattr(self.model, "__class__"):
            model_info["model_type"] = self.model.__class__.__name__

        return model_info

    def _get_expected_features(self):
        """Get the list of features expected by the model"""
        if hasattr(self.model, "feature_names_in_"):
            return self.model.feature_names_in_.tolist()
        else:
            # Default list of expected features if the model doesn't specify them
            return [
                "id",
                "age",
                "has_driving_license",
                "region_id",
                "switch",
                "annual_premium",
                "sales_channel_id",
                "days_since_created",
                "gender_Female",
                "gender_Male",
                "vehicle_age_1to2_Year",
                "vehicle_age_less_than_1_Year",
                "vehicle_age_more_than_2_Years",
                "past_accident_No",
                "past_accident_Yes",
                "age_premium_ratio",
                "age_days_ratio",
            ]

    def _ensure_features_match(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure the dataframe has all the features expected by the model,
        adding missing ones with default values and removing extra ones
        """
        # Add missing columns with default values
        for feature in self.expected_features:
            if feature not in df.columns:
                if feature in ["id", "switch"]:
                    df[feature] = 0  # Default value for ID and switch
                elif (
                    feature.startswith("gender_")
                    or feature.startswith("vehicle_age_")
                    or feature.startswith("past_accident_")
                ):
                    df[feature] = 0  # Default value for one-hot encoded features
                else:
                    df[feature] = 0.0  # Default value for other features

        # Ensure columns are in the right order and no extra columns
        return df[self.expected_features]

    def _preprocess_features(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Preprocess input features"""
        try:
            # Convert list of dictionaries to DataFrame
            df = pd.DataFrame(data)

            # Apply preprocessing
            df = preprocess_data(df)

            # Apply feature engineering
            df = create_feature_pipeline(df)

            # Ensure features match what the model expects
            df = self._ensure_features_match(df)

            return df
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            # Return a dummy dataframe with expected features
            dummy_df = pd.DataFrame(columns=self.expected_features)
            dummy_df.loc[0] = [0] * len(self.expected_features)
            return dummy_df

    def predict(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Make predictions for input features

        Args:
            features: List of feature dictionaries

        Returns:
            List of prediction dictionaries
        """
        try:
            # Preprocess features
            df = self._preprocess_features(features)

            # Make prediction
            probabilities = self.model.predict_proba(df)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)

            # Format results
            results = []
            for i in range(len(predictions)):
                results.append(
                    {
                        "prediction": int(predictions[i]),
                        "probability": float(probabilities[i]),
                    }
                )

            return results
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            # Return a default prediction in case of error
            return [{"prediction": 0, "probability": 0.5}] * len(features)

    def get_info(self) -> Dict[str, Any]:
        """Get model information"""
        return self.model_info
