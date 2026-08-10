"""Pydantic schemas for the DriftGuard inference API."""

from __future__ import annotations

import uuid
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TransactionSchema(BaseModel):
    """Schema representing a single credit card transaction."""

    Time: float = Field(..., description="Seconds elapsed since the reference transaction", ge=0.0)
    Amount: float = Field(..., description="Transaction amount in local currency", ge=0.0)

    # V1 - V28 PCA features
    V1: float = Field(..., description="PCA feature V1")
    V2: float = Field(..., description="PCA feature V2")
    V3: float = Field(..., description="PCA feature V3")
    V4: float = Field(..., description="PCA feature V4")
    V5: float = Field(..., description="PCA feature V5")
    V6: float = Field(..., description="PCA feature V6")
    V7: float = Field(..., description="PCA feature V7")
    V8: float = Field(..., description="PCA feature V8")
    V9: float = Field(..., description="PCA feature V9")
    V10: float = Field(..., description="PCA feature V10")
    V11: float = Field(..., description="PCA feature V11")
    V12: float = Field(..., description="PCA feature V12")
    V13: float = Field(..., description="PCA feature V13")
    V14: float = Field(..., description="PCA feature V14")
    V15: float = Field(..., description="PCA feature V15")
    V16: float = Field(..., description="PCA feature V16")
    V17: float = Field(..., description="PCA feature V17")
    V18: float = Field(..., description="PCA feature V18")
    V19: float = Field(..., description="PCA feature V19")
    V20: float = Field(..., description="PCA feature V20")
    V21: float = Field(..., description="PCA feature V21")
    V22: float = Field(..., description="PCA feature V22")
    V23: float = Field(..., description="PCA feature V23")
    V24: float = Field(..., description="PCA feature V24")
    V25: float = Field(..., description="PCA feature V25")
    V26: float = Field(..., description="PCA feature V26")
    V27: float = Field(..., description="PCA feature V27")
    V28: float = Field(..., description="PCA feature V28")

    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Time": 406.0,
                "Amount": 149.62,
                "V1": -2.31,
                "V2": 1.95,
                "V3": -1.60,
                "V4": 3.99,
                "V5": -0.52,
                "V6": -1.42,
                "V7": -2.53,
                "V8": 1.39,
                "V9": -2.77,
                "V10": -2.77,
                "V11": 3.20,
                "V12": -2.89,
                "V13": -0.59,
                "V14": -4.28,
                "V15": 0.38,
                "V16": -1.14,
                "V17": -2.83,
                "V18": -0.01,
                "V19": 0.41,
                "V20": 0.12,
                "V21": 0.51,
                "V22": -0.03,
                "V23": -0.46,
                "V24": 0.32,
                "V25": 0.04,
                "V26": 0.17,
                "V27": 0.26,
                "V28": -0.14,
            }
        }
    )


class PredictionResponse(BaseModel):
    """Schema for fraud inference response."""

    request_id: str = Field(..., description="Unique trace identifier for the request")
    fraud_score: float = Field(
        ..., description="Estimated probability of fraud (0.0 to 1.0)", ge=0.0, le=1.0
    )
    is_fraud: bool = Field(..., description="Binary classification based on the decision threshold")
    threshold_used: float = Field(..., description="Decision threshold applied")
    model_version: str = Field(..., description="Identifier of the serving model version")
    latency_ms: float = Field(..., description="End-to-end model inference latency in milliseconds")


class BatchTransactionSchema(BaseModel):
    """Schema for batch inference request."""

    transactions: list[TransactionSchema]


class BatchPredictionResponse(BaseModel):
    """Schema for batch inference response."""

    predictions: list[PredictionResponse]
    total_processed: int
    batch_latency_ms: float


class HealthResponse(BaseModel):
    """Schema for service health check."""

    status: str
    model_version: str
    model_loaded: bool
    uptime_seconds: float
    features_count: int


class DriftStatusResponse(BaseModel):
    """Schema for current drift status."""

    drift_detected: bool
    dataset_drift_score: float
    max_feature_psi: float
    drifted_features: list[str]
    sample_window_size: int
    last_evaluated_timestamp: str
