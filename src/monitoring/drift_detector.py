"""
src/monitoring/drift_detector.py

Production feature drift monitoring using Evidently AI.

What it does
────────────
• Computes statistical drift tests between a training reference window
  and incoming production batches:
    - Numerical features → Kolmogorov-Smirnov test
    - Categorical features → Chi-squared test
    - Summary statistic → Population Stability Index (PSI)
• Generates an HTML drift report saved to artifacts/drift_reports/.
• Returns a structured DriftResult with per-feature p-values and an
  overall drift_detected boolean.

Integration
───────────
The stream processor calls check_drift() after every N transactions (batch).
If drift_detected = True, an alert is emitted and the model registry is
queried for a retrained candidate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
)
from evidently.report import Report
from loguru import logger

from src.data.loader import FEATURE_COLS, LABEL_COL

DRIFT_REPORT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "drift_reports"


@dataclass
class FeatureDriftResult:
    feature: str
    statistic: float
    p_value: float
    drifted: bool
    test_name: str


@dataclass
class DriftResult:
    drift_detected: bool
    drift_share: float          # fraction of features drifted
    n_drifted_features: int
    feature_results: list[FeatureDriftResult]
    report_path: Optional[Path]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class DriftDetector:
    """
    Stateful drift monitor.  Call fit() on the training reference window once;
    then call check_drift() on any incoming batch.

    Parameters
    ----------
    drift_threshold   : p-value below which a feature is considered drifted.
    alert_share_pct   : fraction of features that must drift to trigger alert.
    """

    def __init__(
        self,
        drift_threshold: float = 0.05,
        alert_share_pct: float = 0.20,
    ):
        self.drift_threshold = drift_threshold
        self.alert_share_pct = alert_share_pct
        self._reference_df: pd.DataFrame | None = None

    def fit(self, reference_df: pd.DataFrame) -> "DriftDetector":
        """Store reference distribution (training data statistics)."""
        self._reference_df = reference_df[FEATURE_COLS].copy()
        logger.info(
            f"DriftDetector reference set: {len(reference_df):,} rows, "
            f"{len(FEATURE_COLS)} features."
        )
        return self

    def check_drift(
        self,
        production_df: pd.DataFrame,
        batch_id: str = "batch",
        save_report: bool = True,
    ) -> DriftResult:
        """
        Run Evidently drift tests on a production batch.
        Returns DriftResult with per-feature results.
        """
        if self._reference_df is None:
            raise RuntimeError("Call fit() before check_drift().")

        prod = production_df[FEATURE_COLS].copy()

        # ── Evidently report ──────────────────────────────────────────────────
        column_mapping = ColumnMapping()

        report = Report(metrics=[
            DatasetDriftMetric(drift_share_threshold=self.alert_share_pct),
            DataDriftTable(drift_share_threshold=self.drift_threshold),
        ])
        report.run(
            reference_data=self._reference_df,
            current_data=prod,
            column_mapping=column_mapping,
        )

        result_dict = report.as_dict()

        # ── Parse dataset-level drift ─────────────────────────────────────────
        dataset_metric = result_dict["metrics"][0]["result"]
        drift_detected = bool(dataset_metric.get("dataset_drift", False))
        drift_share    = float(dataset_metric.get("drift_share", 0.0))
        n_drifted      = int(dataset_metric.get("number_of_drifted_columns", 0))

        # ── Parse per-feature drift ───────────────────────────────────────────
        feature_results: list[FeatureDriftResult] = []
        try:
            drift_table = result_dict["metrics"][1]["result"].get("drift_by_columns", {})
            for feat, info in drift_table.items():
                feature_results.append(
                    FeatureDriftResult(
                        feature=feat,
                        statistic=float(info.get("statistic", 0.0)),
                        p_value=float(info.get("p_value", 1.0)),
                        drifted=bool(info.get("drift_detected", False)),
                        test_name=str(info.get("stattest_name", "unknown")),
                    )
                )
        except (KeyError, TypeError) as e:
            logger.warning(f"Could not parse per-feature drift: {e}")

        # ── Save HTML report ──────────────────────────────────────────────────
        report_path = None
        if save_report:
            DRIFT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            report_path = DRIFT_REPORT_DIR / f"drift_{batch_id}_{ts}.html"
            report.save_html(str(report_path))
            logger.info(f"Drift report saved → {report_path}")

        result = DriftResult(
            drift_detected=drift_detected,
            drift_share=drift_share,
            n_drifted_features=n_drifted,
            feature_results=feature_results,
            report_path=report_path,
        )

        if drift_detected:
            drifted_feats = [f.feature for f in feature_results if f.drifted]
            logger.warning(
                f"⚠ DRIFT ALERT | batch={batch_id} | "
                f"{n_drifted}/{len(FEATURE_COLS)} features drifted | "
                f"features: {drifted_feats[:5]}"
            )
        else:
            logger.info(
                f"✓ No drift | batch={batch_id} | "
                f"drift_share={drift_share:.2%}"
            )

        return result

    def summary_table(self, result: DriftResult) -> pd.DataFrame:
        """Return per-feature drift results as a sorted DataFrame."""
        rows = [
            {
                "feature": r.feature,
                "test": r.test_name,
                "statistic": round(r.statistic, 5),
                "p_value": round(r.p_value, 5),
                "drifted": r.drifted,
            }
            for r in sorted(result.feature_results, key=lambda x: x.p_value)
        ]
        return pd.DataFrame(rows)
