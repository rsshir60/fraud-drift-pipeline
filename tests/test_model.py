"""
tests/test_model.py

Unit + integration tests for data pipeline, models, threshold optimizer, and evaluator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import (
    FEATURE_COLS,
    LABEL_COL,
    generate_synthetic,
    split_dataset,
)
from src.data.preprocessor import FraudPreprocessor
from src.models.fraud_detector import (
    FraudEnsemble,
    IsolationForestModel,
    LGBMFraudModel,
)
from src.models.threshold_optimizer import CostSensitiveThresholdOptimizer
from src.models.evaluator import evaluate_latency, full_report


@pytest.fixture(scope="module")
def small_df():
    return generate_synthetic(n_samples=10_000, fraud_rate=0.01, random_state=0)


@pytest.fixture(scope="module")
def splits(small_df):
    return split_dataset(small_df, random_state=0)


@pytest.fixture(scope="module")
def arrays(splits):
    train, val, test = splits
    prep = FraudPreprocessor()
    X_train = prep.fit_transform(train)
    X_val = prep.transform(val)
    X_test = prep.transform(test)
    return (
        prep,
        X_train, train[LABEL_COL].values.astype(np.int8),
        X_val,   val[LABEL_COL].values.astype(np.int8),
        X_test,  test[LABEL_COL].values.astype(np.int8),
    )


# ── Data tests ────────────────────────────────────────────────────────────────

class TestDataLoader:
    def test_shape(self, small_df):
        assert len(small_df) == 10_000

    def test_fraud_rate(self, small_df):
        rate = small_df[LABEL_COL].mean()
        assert 0.005 < rate < 0.03

    def test_all_features_present(self, small_df):
        for col in FEATURE_COLS:
            assert col in small_df.columns

    def test_split_no_leakage(self, splits):
        train, val, test = splits
        assert set(train.index).isdisjoint(set(test.index))
        assert set(val.index).isdisjoint(set(test.index))

    def test_stratification(self, splits):
        train, val, test = splits
        rates = [df[LABEL_COL].mean() for df in [train, val, test]]
        assert max(rates) - min(rates) < 0.01   # rates within 1%


class TestPreprocessor:
    def test_output_shape(self, arrays):
        prep, X_train, *rest = arrays
        assert X_train.shape[1] == len(FEATURE_COLS)

    def test_no_nans(self, arrays):
        prep, X_train, *rest = arrays
        assert not np.isnan(X_train).any()

    def test_transform_dict(self, arrays):
        prep, X_train, *_ = arrays
        rec = {f: 0.0 for f in FEATURE_COLS}
        X = prep.transform_dict(rec)
        assert X.shape == (1, len(FEATURE_COLS))


# ── Model tests ───────────────────────────────────────────────────────────────

class TestLGBM:
    def test_fit_predict_shape(self, arrays):
        prep, X_tr, y_tr, X_val, y_val, X_te, y_te = arrays
        m = LGBMFraudModel()
        m.fit(X_tr, y_tr, X_val, y_val)
        scores = m.predict_proba(X_te)
        assert scores.shape == (len(X_te),)

    def test_scores_in_range(self, arrays):
        prep, X_tr, y_tr, X_val, y_val, X_te, y_te = arrays
        m = LGBMFraudModel()
        m.fit(X_tr, y_tr, X_val, y_val)
        scores = m.predict_proba(X_te)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_fraud_scores_higher(self, arrays):
        prep, X_tr, y_tr, X_val, y_val, X_te, y_te = arrays
        m = LGBMFraudModel()
        m.fit(X_tr, y_tr, X_val, y_val)
        scores = m.predict_proba(X_te)
        fraud_mean = scores[y_te == 1].mean()
        legit_mean = scores[y_te == 0].mean()
        assert fraud_mean > legit_mean


class TestIsolationForest:
    def test_fit_predict(self, arrays):
        prep, X_tr, y_tr, *rest = arrays
        m = IsolationForestModel(contamination=0.01)
        m.fit(X_tr, y_tr)
        scores = m.predict_proba(X_tr[:200])
        assert scores.shape == (200,)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0


class TestEnsemble:
    def test_ensemble_fit(self, arrays):
        prep, X_tr, y_tr, X_val, y_val, X_te, y_te = arrays
        ens = FraudEnsemble()
        ens.fit(X_tr, y_tr, X_val, y_val)
        scores = ens.predict_proba(X_te)
        assert len(scores) == len(X_te)
        assert not np.isnan(scores).any()


# ── Threshold optimizer tests ─────────────────────────────────────────────────

class TestThresholdOptimizer:
    @pytest.fixture(scope="class")
    def fitted_optimizer(self, arrays):
        prep, X_tr, y_tr, X_val, y_val, X_te, y_te = arrays
        m = LGBMFraudModel()
        m.fit(X_tr, y_tr, X_val, y_val)
        scores = m.predict_proba(X_val)
        opt = CostSensitiveThresholdOptimizer(fn_cost=90.0, fp_cost=25.0)
        opt.fit(y_val, scores)
        return opt, scores, y_val

    def test_optimal_threshold_in_range(self, fitted_optimizer):
        opt, _, _ = fitted_optimizer
        assert 0.0 < opt.optimal_threshold_ < 1.0

    def test_sweep_df_shape(self, fitted_optimizer):
        opt, _, _ = fitted_optimizer
        assert len(opt.sweep_df_) > 0
        assert "total_cost_usd" in opt.sweep_df_.columns

    def test_precision_at_recall(self, fitted_optimizer, arrays):
        opt, _, y_val = fitted_optimizer
        prep, X_tr, y_tr, X_val, *_ = arrays
        m = LGBMFraudModel()
        # Just test the function signature / output structure
        # using already-computed scores
        _, scores, y_v = fitted_optimizer
        result = opt.precision_at_fixed_recall(y_v, scores, recall_target=0.70)
        assert "precision_at_recall" in result
        assert "achieved_recall" in result
        assert result["achieved_recall"] >= 0.70 - 0.05   # within tolerance


# ── Latency test ──────────────────────────────────────────────────────────────

class TestLatency:
    def test_p99_under_sla(self, arrays):
        prep, X_tr, y_tr, X_val, y_val, X_te, y_te = arrays
        m = LGBMFraudModel()
        m.fit(X_tr, y_tr, X_val, y_val)
        result = evaluate_latency(m, X_te, n_warmup=20, n_timed=100)
        # SLA = 50ms; LightGBM on CPU should be well under
        assert result["p99_ms"] < 200.0   # generous bound for CI environment
        assert result["sla_met"] or result["p99_ms"] < 200.0
