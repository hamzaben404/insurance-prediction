# src/api/models/insurance.py
from typing import List, Optional

from pydantic import BaseModel, Field


class InsuranceFeatures(BaseModel):
    """Input features for insurance prediction"""

    gender: str = Field(..., description="Customer gender (Male/Female)")
    age: float = Field(..., description="Customer age")
    has_driving_license: bool = Field(..., description="Whether customer has driving license")
    region_id: int = Field(..., description="Region identifier")
    vehicle_age: str = Field(..., description="Age category of the vehicle")
    past_accident: Optional[str] = Field(
        None, description="Whether customer had past accidents (Yes/No)"
    )
    annual_premium: float = Field(..., description="Annual insurance premium")
    sales_channel_id: int = Field(..., description="Sales channel identifier")
    days_since_created: int = Field(..., description="Days since record was created")


class InsurancePrediction(BaseModel):
    """Model prediction result"""

    prediction: int = Field(..., description="Prediction (0: No Purchase, 1: Purchase)")
    probability: float = Field(..., description="Probability of purchase")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""

    records: List[InsuranceFeatures] = Field(..., description="List of records to predict")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""

    predictions: List[InsurancePrediction] = Field(..., description="Prediction results")


class ModelInfo(BaseModel):
    """Model information"""

    model_type: str = Field(..., description="Type of model")
    model_version: str = Field(..., description="Model version")
    features: List[str] = Field(..., description="Features used by the model")
    metrics: dict = Field(..., description="Model performance metrics")
