"""
scripts/simulate_stream.py

Replay synthetic transactions through the full inference + drift pipeline
without a live Kafka broker.

Usage
─────
  python scripts/simulate_stream.py
  python scripts/simulate_stream.py --n-transactions 10000 --inject-drift
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import typer
from loguru import logger

from src.data.loader import load_dataset, split_dataset
from src.data.preprocessor import FraudPreprocessor
from src.models.fraud_detector import FraudEnsemble
from src.monitoring.drift_detector import DriftDetector
from src.pipeline.stream_processor import FraudStreamProcessor

ARTIFACT_DIR = ROOT / "artifacts" / "models"
app = typer.Typer(add_completion=False)


@app.command()
def simulate(
    n_transactions: int = typer.Option(5000, help="Number of transactions to replay"),
    inject_drift: bool = typer.Option(True, help="Inject a distribution shift mid-stream"),
    random_state: int = typer.Option(42),
):
    # ── Load artefacts ────────────────────────────────────────────────────────
    for name in ["preprocessor.joblib", "fraud_ensemble.joblib", "drift_detector.joblib"]:
        if not (ARTIFACT_DIR / name).exists():
            logger.error(f"{name} missing. Run scripts/train.py first.")
            raise typer.Exit(1)

    prep = FraudPreprocessor.load(ARTIFACT_DIR / "preprocessor.joblib")
    model = FraudEnsemble.load(ARTIFACT_DIR / "fraud_ensemble.joblib")
    drift_det: DriftDetector = joblib.load(ARTIFACT_DIR / "drift_detector.joblib")
    threshold = float(np.load(ARTIFACT_DIR / "optimal_threshold.npy"))

    # ── Load stream data ──────────────────────────────────────────────────────
    df = load_dataset(source="synthetic", n_samples=max(n_transactions + 1000, 20_000), random_state=random_state)
    _, _, stream_df = split_dataset(df, random_state=random_state)

    # ── Build processor ───────────────────────────────────────────────────────
    processor = FraudStreamProcessor(
        model=model,
        preprocessor=prep,
        drift_detector=drift_det,
        threshold=threshold,
        drift_batch_size=500,
    )

    # ── Run simulation ────────────────────────────────────────────────────────
    inject_at = n_transactions // 2 if inject_drift else None
    summary = processor.simulate_stream(
        df=stream_df,
        n_transactions=n_transactions,
        inject_drift_at=inject_at,
    )

    print("\n══ STREAM SIMULATION RESULTS ══════════════════════")
    for k, v in summary.items():
        print(f"  {k:<28}: {v}")
    print("═══════════════════════════════════════════════════\n")


if __name__ == "__main__":
    app()
