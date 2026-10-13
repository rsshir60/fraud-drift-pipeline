"""
src/pipeline/stream_processor.py

Kafka consumer pipeline:

  Kafka topic (transactions) → decode → preprocess → score → flag → produce alert

Architecture
────────────
• Consumes from topic `transactions` — each message is a JSON transaction record.
• Buffers messages into micro-batches (default 100) for drift monitoring.
• Every micro-batch: run drift check, emit alert if drift detected.
• Per-transaction: run fraud scoring < SLA (50ms), produce result to `fraud_scores`.
• Structured logging in JSON format for downstream log aggregation (Datadog/Grafana).

Without Kafka running
─────────────────────
Use `simulate_stream()` which replays synthetic data through the same
inference stack without a broker — useful for demos and latency benchmarking.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from loguru import logger

from src.data.loader import FEATURE_COLS
from src.data.preprocessor import FraudPreprocessor
from src.models.fraud_detector import FraudEnsemble
from src.monitoring.drift_detector import DriftDetector, DriftResult

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_IN  = os.getenv("KAFKA_TOPIC_IN",  "transactions")
TOPIC_OUT = os.getenv("KAFKA_TOPIC_OUT", "fraud_scores")
TOPIC_ALERTS = os.getenv("KAFKA_TOPIC_ALERTS", "drift_alerts")
DRIFT_BATCH_SIZE = int(os.getenv("DRIFT_BATCH_SIZE", "500"))


class FraudStreamProcessor:
    """
    Stateful stream processor. Holds all loaded artefacts in memory.

    Parameters
    ----------
    model         : Trained FraudEnsemble.
    preprocessor  : Fitted FraudPreprocessor.
    drift_detector: Fitted DriftDetector with training reference.
    threshold     : Decision threshold (cost-optimised).
    """

    def __init__(
        self,
        model: FraudEnsemble,
        preprocessor: FraudPreprocessor,
        drift_detector: DriftDetector,
        threshold: float = 0.30,
        drift_batch_size: int = DRIFT_BATCH_SIZE,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.drift_detector = drift_detector
        self.threshold = threshold
        self.drift_batch_size = drift_batch_size

        self._buffer: list[dict] = []
        self._total_processed = 0
        self._total_flagged = 0
        self._drift_alerts = 0

    def score_transaction(self, record: dict) -> dict:
        """
        Score a single transaction dict.
        Returns enriched record with fraud_score and is_flagged fields.
        """
        t0 = time.perf_counter()
        X = self.preprocessor.transform_dict(record)
        fraud_score = float(self.model.predict_proba(X)[0])
        latency_ms = (time.perf_counter() - t0) * 1000

        is_flagged = fraud_score >= self.threshold
        self._total_processed += 1
        if is_flagged:
            self._total_flagged += 1

        return {
            "transaction_id": record.get("transaction_id", "UNKNOWN"),
            "fraud_score": round(fraud_score, 6),
            "is_flagged": is_flagged,
            "threshold": self.threshold,
            "latency_ms": round(latency_ms, 3),
        }

    def _check_batch_drift(self, batch_id: str) -> DriftResult | None:
        """Run drift check on the accumulated buffer."""
        if len(self._buffer) < self.drift_batch_size:
            return None
        df = pd.DataFrame(self._buffer[-self.drift_batch_size :])
        # Fill any missing feature columns with zeros
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
        result = self.drift_detector.check_drift(df, batch_id=batch_id)
        if result.drift_detected:
            self._drift_alerts += 1
        return result

    def stats(self) -> dict:
        flag_rate = (
            self._total_flagged / self._total_processed
            if self._total_processed > 0
            else 0.0
        )
        return {
            "total_processed": self._total_processed,
            "total_flagged": self._total_flagged,
            "flag_rate_pct": round(flag_rate * 100, 3),
            "drift_alerts_fired": self._drift_alerts,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Kafka consumer loop
    # ──────────────────────────────────────────────────────────────────────────

    def run_kafka(self) -> None:
        """
        Blocking consumer loop. Requires a running Kafka/Redpanda broker.
        Start with: docker compose up kafka
        """
        try:
            from kafka import KafkaConsumer, KafkaProducer
        except ImportError:
            raise ImportError("kafka-python not installed. pip install kafka-python")

        consumer = KafkaConsumer(
            TOPIC_IN,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="latest",
            group_id="fraud-detector",
            enable_auto_commit=True,
        )
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

        logger.info(
            f"Kafka consumer started | topic={TOPIC_IN} | "
            f"broker={KAFKA_BOOTSTRAP}"
        )

        batch_count = 0
        for msg in consumer:
            record = msg.value
            result = self.score_transaction(record)
            self._buffer.append(record)

            # Produce score back to Kafka
            producer.send(TOPIC_OUT, value=result)

            # Drift check every N messages
            if len(self._buffer) % self.drift_batch_size == 0:
                batch_count += 1
                drift_result = self._check_batch_drift(f"b{batch_count}")
                if drift_result and drift_result.drift_detected:
                    alert = {
                        "event": "DRIFT_DETECTED",
                        "batch_id": f"b{batch_count}",
                        "drift_share": drift_result.drift_share,
                        "n_drifted_features": drift_result.n_drifted_features,
                        "timestamp": drift_result.timestamp,
                    }
                    producer.send(TOPIC_ALERTS, value=alert)
                    logger.warning(f"Drift alert produced → {TOPIC_ALERTS}")

            if self._total_processed % 1000 == 0:
                logger.info(f"Stats: {self.stats()}")

    # ──────────────────────────────────────────────────────────────────────────
    # Simulation (no broker needed)
    # ──────────────────────────────────────────────────────────────────────────

    def simulate_stream(
        self,
        df: pd.DataFrame,
        n_transactions: int = 5000,
        inject_drift_at: int | None = 2500,
    ) -> dict:
        """
        Replay a DataFrame through the inference stack without Kafka.
        Optionally inject a simulated drift event at `inject_drift_at`.

        Returns performance summary dict.
        """
        from src.data.loader import generate_synthetic

        latencies: list[float] = []
        n = min(n_transactions, len(df))
        logger.info(f"Simulating {n:,} transactions through pipeline …")

        for i in range(n):
            row = df.iloc[i].to_dict()

            # Inject drift: swap in a drifted batch at mid-point
            if inject_drift_at and i == inject_drift_at:
                logger.info("Injecting distribution drift …")
                drifted_df = generate_synthetic(
                    n_samples=self.drift_batch_size * 2,
                    drift=True,
                    random_state=999,
                )
                drift_result = self.drift_detector.check_drift(
                    drifted_df, batch_id="injected_drift", save_report=True
                )
                if drift_result.drift_detected:
                    logger.warning("Drift injection confirmed detected ✓")

            result = self.score_transaction(row)
            self._buffer.append(row)
            latencies.append(result["latency_ms"])

            # Periodic drift check
            if len(self._buffer) % self.drift_batch_size == 0:
                batch_id = f"sim_b{len(self._buffer) // self.drift_batch_size}"
                self._check_batch_drift(batch_id)

        latency_summary = {
            "p50_ms": round(float(np.percentile(latencies, 50)), 3),
            "p95_ms": round(float(np.percentile(latencies, 95)), 3),
            "p99_ms": round(float(np.percentile(latencies, 99)), 3),
            "mean_ms": round(float(np.mean(latencies)), 3),
            "sla_met_pct": round(
                sum(1 for l in latencies if l < 50) / len(latencies) * 100, 2
            ),
        }
        summary = {**self.stats(), **latency_summary}
        logger.info("Simulation complete:\n" + "\n".join(f"  {k}: {v}" for k, v in summary.items()))
        return summary
