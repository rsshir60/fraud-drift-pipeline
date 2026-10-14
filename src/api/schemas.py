"""
src/api/schemas.py

Pydantic v2 request/response models for the fraud detection API.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    transaction_id: str = Field(default="", description="Optional ID for tracking")
    transaction_amount: float = Field(..., gt=0, description="Transaction amount in USD")
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    is_international: int = Field(..., ge=0, le=1)
    merchant_category_encoded: int = Field(..., ge=0, le=9)
    card_type_encoded: int = Field(..., ge=0, le=2, description="0=credit 1=debit 2=prepaid")
    distance_from_home_km: float = Field(..., ge=0)
    time_since_last_txn_min: float = Field(..., ge=0)
    n_txn_last_1h: int = Field(..., ge=0)
    n_txn_last_24h: int = Field(..., ge=0)
    n_txn_last_7d: int = Field(..., ge=0)
    avg_amount_last_30d: float = Field(..., ge=0)
    std_amount_last_30d: float = Field(..., ge=0)
    amount_vs_avg_ratio: float = Field(default=1.0)
    card_age_days: int = Field(..., ge=0)
    account_age_days: int = Field(..., ge=0)
    is_weekend: int = Field(default=0, ge=0, le=1)
    is_night: int = Field(default=0, ge=0, le=1)
    velocity_score: float = Field(default=0.0, ge=0)
    amount_log: float = Field(default=0.0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_id": "TXN-00012345",
                "transaction_amount": 4500.00,
                "hour_of_day": 3,
                "day_of_week": 1,
                "is_international": 1,
                "merchant_category_encoded": 9,
                "card_type_encoded": 2,
                "distance_from_home_km": 850.0,
                "time_since_last_txn_min": 2.5,
                "n_txn_last_1h": 8,
                "n_txn_last_24h": 22,
                "n_txn_last_7d": 45,
                "avg_amount_last_30d": 95.0,
                "std_amount_last_30d": 30.0,
                "amount_vs_avg_ratio": 47.4,
                "card_age_days": 45,
                "account_age_days": 12,
                "is_weekend": 0,
                "is_night": 1,
                "velocity_score": 28.0,
                "amount_log": 8.41,
            }
        }
    }


class BatchScoreRequest(BaseModel):
    transactions: list[TransactionRequest] = Field(
        ..., min_length=1, max_length=500
    )


class FraudScoreResponse(BaseModel):
    transaction_id: str
    fraud_score: float = Field(..., description="Fraud probability [0, 1]")
    is_flagged: bool = Field(..., description="True if score >= threshold")
    threshold: float
    risk_level: str = Field(..., description="LOW | MEDIUM | HIGH | CRITICAL")
    latency_ms: float


class BatchScoreResponse(BaseModel):
    n_transactions: int
    n_flagged: int
    flag_rate_pct: float
    threshold: float
    p99_latency_ms: float
    sla_met: bool
    scores: list[FraudScoreResponse]


class DriftStatusResponse(BaseModel):
    drift_detected: bool
    drift_share_pct: float
    n_drifted_features: int
    top_drifted_features: list[str]
    alert_message: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model: str
    threshold: float
    version: str
