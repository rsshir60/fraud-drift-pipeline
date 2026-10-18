"""
tests/test_api.py

Async integration tests for the FastAPI fraud detection service.
"""

from __future__ import annotations

import numpy as np
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock

from src.monitoring.drift_detector import DriftResult, FeatureDriftResult

SAMPLE_TX = {
    "transaction_id": "TXN-TEST-001",
    "transaction_amount": 4500.0,
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

LEGIT_TX = {**SAMPLE_TX, "transaction_id": "TXN-LEGIT", "transaction_amount": 45.0,
            "hour_of_day": 14, "is_international": 0, "n_txn_last_1h": 1,
            "distance_from_home_km": 2.0, "velocity_score": 2.0, "amount_log": 3.8}


def _mock_score_single(record):
    amount = record.get("transaction_amount", 0)
    score = 0.85 if amount > 1000 else 0.05
    return {"transaction_id": record.get("transaction_id", ""), "fraud_score": score,
            "is_flagged": score >= 0.30, "threshold": 0.30, "latency_ms": 1.2}


def _mock_score_batch(records):
    results = [_mock_score_single(r) for r in records]
    latencies = [r["latency_ms"] for r in results]
    return results, latencies


def _mock_drift_result():
    return DriftResult(
        drift_detected=True,
        drift_share=0.35,
        n_drifted_features=7,
        feature_results=[
            FeatureDriftResult("transaction_amount", 0.45, 0.001, True, "ks"),
            FeatureDriftResult("velocity_score", 0.32, 0.01, True, "ks"),
        ],
        report_path=None,
    )


@pytest.fixture(scope="module")
def client():
    with (
        patch("src.api.predictor.load_all_artifacts", return_value=None),
        patch("src.api.predictor._state", {"threshold": 0.30, "model_name": "fraud_ensemble"}),
        patch("src.api.predictor.score_single", side_effect=_mock_score_single),
        patch("src.api.predictor.score_batch_internal", side_effect=_mock_score_batch),
        patch("src.api.predictor.run_drift_check", return_value=_mock_drift_result()),
        patch("src.api.predictor.get_drift_status", return_value=_mock_drift_result()),
        patch("src.api.predictor._counters", {"total_scored": 0, "total_flagged": 0,
                                               "drift_checks_run": 0, "drift_alerts": 0}),
    ):
        from src.api.main import app
        transport = ASGITransport(app=app)
        c = AsyncClient(transport=transport, base_url="http://test")
        return c


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_score_high_risk(client):
    resp = await client.post("/score", json=SAMPLE_TX)
    assert resp.status_code == 200
    body = resp.json()
    assert body["fraud_score"] > 0.5
    assert body["is_flagged"] is True
    assert body["risk_level"] in ("HIGH", "CRITICAL")


@pytest.mark.asyncio
async def test_score_low_risk(client):
    resp = await client.post("/score", json=LEGIT_TX)
    assert resp.status_code == 200
    body = resp.json()
    assert body["fraud_score"] < 0.3
    assert body["is_flagged"] is False
    assert body["risk_level"] == "LOW"


@pytest.mark.asyncio
async def test_batch_score(client):
    payload = {"transactions": [SAMPLE_TX, LEGIT_TX, SAMPLE_TX]}
    resp = await client.post("/score/batch", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_transactions"] == 3
    assert body["n_flagged"] == 2
    assert "p99_latency_ms" in body
    assert "sla_met" in body


@pytest.mark.asyncio
async def test_batch_size_limit(client):
    payload = {"transactions": [SAMPLE_TX] * 501}
    resp = await client.post("/score/batch", json=payload)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_drift_check(client):
    payload = {"transactions": [SAMPLE_TX] * 10}
    resp = await client.post("/drift/check", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "drift_detected" in body
    assert "n_drifted_features" in body


@pytest.mark.asyncio
async def test_drift_latest(client):
    resp = await client.get("/drift/latest")
    assert resp.status_code == 200
    body = resp.json()
    assert body["drift_detected"] is True
    assert len(body["top_drifted_features"]) > 0
    assert body["alert_message"] is not None


@pytest.mark.asyncio
async def test_metrics_endpoint(client):
    resp = await client.get("/metrics")
    assert resp.status_code == 200
