# src/api/models/insurance.py
from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Request model for insurance prediction"""

    gender: str = Field(..., description="Gender of the person (Male/Female)")
    age: float = Field(..., ge=18, le=100, description="Age of the person")
    has_driving_license: bool = Field(..., description="Whether person has driving license")
    region_id: int = Field(..., description="Region ID")
    vehicle_age: str = Field(..., description="Age of vehicle")
    past_accident: str = Field(..., description="Whether person had past accidents (Yes/No)")
    annual_premium: float = Field(..., gt=0, description="Annual premium amount")
    sales_channel_id: int = Field(..., description="Sales channel ID")
    days_since_created: int = Field(..., ge=0, description="Days since policy created")


class PredictionResponse(BaseModel):
    """Response model for insurance prediction"""

    prediction: int = Field(..., description="Prediction result (0 or 1)")
    probability: float = Field(..., ge=0, le=1, description="Probability of positive class")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""

    records: list[PredictionRequest] = Field(..., description="List of records to predict")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""

    predictions: list[PredictionResponse] = Field(..., description="List of predictions")


class ModelInfo(BaseModel):
    """Model information"""

    model_config = ConfigDict(protected_namespaces=())  # Fix the warning

    model_type: str = Field(..., description="Type of model")
    model_version: str = Field(..., description="Version of model")
    features: list[str] = Field(default_factory=list, description="List of features")
    metrics: dict = Field(default_factory=dict, description="Model metrics")
