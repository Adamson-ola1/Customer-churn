"""
backend/schema.py
Pydantic request/response models for the FastAPI churn-prediction service.
"""
from typing import Literal

from pydantic import BaseModel, Field


class CustomerRecord(BaseModel):
    """Raw customer attributes required by the model (pre-feature-engineering)."""
    credit_score: int = Field(..., ge=300, le=900, example=619)
    country: Literal["France", "Spain", "Germany"] = Field(..., example="France")
    gender: Literal["Male", "Female"] = Field(..., example="Female")
    age: int = Field(..., ge=18, le=100, example=42)
    tenure: int = Field(..., ge=0, le=15, example=2)
    balance: float = Field(..., ge=0, example=0.0)
    products_number: int = Field(..., ge=1, le=4, example=1)
    credit_card: Literal[0, 1] = Field(..., example=1)
    active_member: Literal[0, 1] = Field(..., example=1)
    estimated_salary: float = Field(..., ge=0, example=101348.88)


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[dict]


class ModelMetric(BaseModel):
    model: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float


class ModelInfoResponse(BaseModel):
    best_model: str
    results: list[dict]
    features: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    
