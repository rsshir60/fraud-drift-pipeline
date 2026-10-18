"""
tests/test_drift.py

Unit and integration tests for Evidently AI drift detection suite.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.loader import generate_synthetic, FEATURE_COLS
from src.monitoring.drift_detector import DriftDetector, DriftResult


@pytest.fixture(scope="module")
def baseline_data():
    return generate_synthetic(n_samples=2000, drift=False, random_state=42)


@pytest.fixture(scope="module")
def drifted_data():
    return generate_synthetic(n_samples=1000, drift=True, random_state=999)


@pytest.fixture(scope="module")
def non_drifted_data():
    return generate_synthetic(n_samples=1000, drift=False, random_state=123)


class TestDriftDetector:
    def test_fit_stores_reference(self, baseline_data):
        detector = DriftDetector(drift_threshold=0.05, alert_share_pct=0.20)
        detector.fit(baseline_data)
        assert detector._reference_df is not None
        assert len(detector._reference_df) == len(baseline_data)
        assert list(detector._reference_df.columns) == FEATURE_COLS

    def test_check_drift_no_drift_detected(self, baseline_data, non_drifted_data):
        detector = DriftDetector(drift_threshold=0.01, alert_share_pct=0.50)
        detector.fit(baseline_data)
        result = detector.check_drift(non_drifted_data, batch_id="clean_test", save_report=False)
        assert isinstance(result, DriftResult)
        assert not result.drift_detected
        assert result.drift_share < 0.50

    def test_check_drift_detected_on_shifted_distribution(self, baseline_data, drifted_data):
        detector = DriftDetector(drift_threshold=0.05, alert_share_pct=0.15)
        detector.fit(baseline_data)
        result = detector.check_drift(drifted_data, batch_id="drifted_test", save_report=False)
        assert isinstance(result, DriftResult)
        assert result.drift_detected is True
        assert result.n_drifted_features > 0
        assert result.drift_share >= 0.15

    def test_summary_table(self, baseline_data, drifted_data):
        detector = DriftDetector()
        detector.fit(baseline_data)
        result = detector.check_drift(drifted_data, batch_id="table_test", save_report=False)
        table = detector.summary_table(result)
        assert isinstance(table, pd.DataFrame)
        assert "feature" in table.columns
        assert "p_value" in table.columns
        assert "drifted" in table.columns
        assert len(table) > 0
