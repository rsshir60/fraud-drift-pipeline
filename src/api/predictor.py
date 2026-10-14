"""
src/api/predictor.py

Module-level artefact cache + inference functions for the FastAPI layer.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.data.preprocessor import FraudPreprocessor
from src.models.fraud_detector import FraudEnsemble
from src.monitoring.drift_detector import DriftDetector, DriftResult

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "models"

# ── Module-level state ────────────────────────────────────────────────────────
_model: Optional[FraudEnsemble] = None
_preprocessor: Optional[FraudPreprocessor] = None
_drift_detector: Optional[DriftDetector] = None
_last_drift_result: Optional[DriftResult] = None
_state: dict = {}
_counters: dict = {
    "total_scored": 0,
    "total_flagged": 0,
    "drift_checks_run": 0,
    "drift_alerts": 0,
}


def load_all_artifacts() -> None:
    global _model, _preprocessor, _drift_detector, _state

    prep_path = ARTIFACT_DIR / "preprocessor.joblib"
    model_path = ARTIFACT_DIR / "fraud_ensemble.joblib"
    drift_path = ARTIFACT_DIR / "drift_detector.joblib"
    threshold_path = ARTIFACT_DIR / "optimal_threshold.npy"

    for p in [prep_path, model_path]:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run scripts/train.py first.")

    _preprocessor = FraudPreprocessor.load(prep_path)
    _model = FraudEnsemble.load(model_path)

    threshold = float(np.load(threshold_path)) if threshold_path.exists() else 0.30
    _state["threshold"] = threshold
    _state["model_name"] = "fraud_ensemble"

    if drift_path.exists():
        import joblib
        _drift_detector = joblib.load(drift_path)
        logger.info("DriftDetector loaded.")
    else:
        logger.warning("DriftDetector not found — drift endpoints will be unavailable.")

    logger.info(f"Artefacts loaded | threshold={threshold:.4f}")


def score_single(record: dict) -> dict:
    global _counters
    t0 = time.perf_counter()
    X = _preprocessor.transform_dict(record)
    score = float(_model.predict_proba(X)[0])
    latency_ms = (time.perf_counter() - t0) * 1000

    threshold = _state["threshold"]
    is_flagged = score >= threshold

    _counters["total_scored"] += 1
    if is_flagged:
        _counters["total_flagged"] += 1

    return {
        "transaction_id": record.get("transaction_id", ""),
        "fraud_score": round(score, 6),
        "is_flagged": is_flagged,
        "threshold": threshold,
        "latency_ms": round(latency_ms, 3),
    }


def score_batch_internal(records: list[dict]) -> tuple[list[dict], list[float]]:
    threshold = _state["threshold"]
    results = []
    latencies = []

    for record in records:
        t0 = time.perf_counter()
        X = _preprocessor.transform_dict(record)
        score = float(_model.predict_proba(X)[0])
        latency_ms = (time.perf_counter() - t0) * 1000

        latencies.append(latency_ms)
        is_flagged = score >= threshold
        _counters["total_scored"] += 1
        if is_flagged:
            _counters["total_flagged"] += 1

        results.append({
            "transaction_id": record.get("transaction_id", ""),
            "fraud_score": round(score, 6),
            "is_flagged": is_flagged,
            "threshold": threshold,
            "latency_ms": round(latency_ms, 3),
        })

    return results, latencies


def run_drift_check(records: list[dict], batch_id: str = "api") -> DriftResult:
    global _last_drift_result
    if _drift_detector is None:
        raise RuntimeError("DriftDetector not loaded.")
    df = pd.DataFrame(records)
    result = _drift_detector.check_drift(df, batch_id=batch_id, save_report=True)
    _last_drift_result = result
    _counters["drift_checks_run"] += 1
    if result.drift_detected:
        _counters["drift_alerts"] += 1
    return result


def get_drift_status() -> Optional[DriftResult]:
    return _last_drift_result
