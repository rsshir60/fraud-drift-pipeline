"""
src/api/main.py

FastAPI fraud detection service.

Endpoints
─────────
GET  /health              → liveness + threshold
POST /score               → single transaction scoring
POST /score/batch         → batch scoring (up to 500), returns P99 latency
POST /drift/check         → run drift check on a submitted batch
GET  /drift/latest        → last drift check result
GET  /metrics             → running counters (flagged, processed, drift alerts)
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.predictor import load_all_artifacts, score_single, score_batch_internal, get_drift_status, run_drift_check
from src.api.schemas import (
    BatchScoreRequest,
    BatchScoreResponse,
    DriftStatusResponse,
    FraudScoreResponse,
    HealthResponse,
    TransactionRequest,
)

APP_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading fraud detection artefacts …")
    load_all_artifacts()
    logger.info("API ready.")
    yield
    logger.info("Shutdown.")


app = FastAPI(
    title="Fraud & Drift Detection API",
    description=(
        "Real-time transaction fraud scoring with asymmetric cost-optimised "
        "threshold and automated feature drift monitoring via Evidently AI."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _risk_level(score: float) -> str:
    if score < 0.15:  return "LOW"
    if score < 0.40:  return "MEDIUM"
    if score < 0.70:  return "HIGH"
    return "CRITICAL"


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    from src.api.predictor import _state
    return HealthResponse(
        status="ok",
        model=_state.get("model_name", "fraud_ensemble"),
        threshold=_state.get("threshold", 0.0),
        version=APP_VERSION,
    )


@app.post("/score", response_model=FraudScoreResponse, tags=["inference"])
async def score_transaction(tx: TransactionRequest):
    """Score a single transaction. P99 latency target: < 50ms."""
    try:
        result = score_single(tx.model_dump())
    except Exception as e:
        logger.exception("Scoring failed")
        raise HTTPException(status_code=500, detail=str(e))

    return FraudScoreResponse(
        transaction_id=tx.transaction_id or "N/A",
        fraud_score=result["fraud_score"],
        is_flagged=result["is_flagged"],
        threshold=result["threshold"],
        risk_level=_risk_level(result["fraud_score"]),
        latency_ms=result["latency_ms"],
    )


@app.post("/score/batch", response_model=BatchScoreResponse, tags=["inference"])
async def score_batch(request: BatchScoreRequest):
    """
    Score a batch of transactions. Returns per-transaction scores plus
    P99 latency and SLA compliance flag (< 50ms).
    """
    records = [t.model_dump() for t in request.transactions]
    try:
        results, latencies = score_batch_internal(records)
    except Exception as e:
        logger.exception("Batch scoring failed")
        raise HTTPException(status_code=500, detail=str(e))

    p99 = float(np.percentile(latencies, 99))
    scores = [
        FraudScoreResponse(
            transaction_id=r.get("transaction_id", f"idx_{i}") or f"idx_{i}",
            fraud_score=r["fraud_score"],
            is_flagged=r["is_flagged"],
            threshold=r["threshold"],
            risk_level=_risk_level(r["fraud_score"]),
            latency_ms=r["latency_ms"],
        )
        for i, r in enumerate(results)
    ]
    n_flagged = sum(1 for s in scores if s.is_flagged)

    return BatchScoreResponse(
        n_transactions=len(scores),
        n_flagged=n_flagged,
        flag_rate_pct=round(n_flagged / len(scores) * 100, 2),
        threshold=scores[0].threshold,
        p99_latency_ms=round(p99, 3),
        sla_met=p99 < 50.0,
        scores=scores,
    )


@app.post("/drift/check", response_model=DriftStatusResponse, tags=["monitoring"])
async def check_drift(request: BatchScoreRequest):
    """
    Run drift detection on a submitted batch of transactions.
    The batch is compared against the training reference distribution.
    """
    records = [t.model_dump() for t in request.transactions]
    try:
        drift_result = run_drift_check(records)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    top_drifted = [
        f.feature for f in drift_result.feature_results if f.drifted
    ][:5]

    return DriftStatusResponse(
        drift_detected=drift_result.drift_detected,
        drift_share_pct=round(drift_result.drift_share * 100, 1),
        n_drifted_features=drift_result.n_drifted_features,
        top_drifted_features=top_drifted,
        alert_message=(
            f"⚠ Drift detected in {drift_result.n_drifted_features} features. "
            f"Consider retraining."
            if drift_result.drift_detected
            else None
        ),
    )


@app.get("/drift/latest", response_model=DriftStatusResponse, tags=["monitoring"])
async def latest_drift():
    """Return the most recent drift check result."""
    result = get_drift_status()
    if result is None:
        raise HTTPException(status_code=404, detail="No drift check run yet.")
    top_drifted = [f.feature for f in result.feature_results if f.drifted][:5]
    return DriftStatusResponse(
        drift_detected=result.drift_detected,
        drift_share_pct=round(result.drift_share * 100, 1),
        n_drifted_features=result.n_drifted_features,
        top_drifted_features=top_drifted,
    )


@app.get("/metrics", tags=["ops"])
async def metrics():
    from src.api.predictor import _counters
    return _counters
