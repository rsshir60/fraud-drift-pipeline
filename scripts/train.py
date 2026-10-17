"""
scripts/train.py

End-to-end training pipeline:
  1. Generate / load dataset
  2. Preprocess (RobustScaler)
  3. Train FraudEnsemble (LightGBM + IsolationForest)
  4. Cost-sensitive threshold optimisation
  5. Evaluate: Precision@Recall, P99 latency, confusion matrix
  6. Fit DriftDetector on training reference window
  7. Log everything to MLflow

Usage
─────
  python scripts/train.py
  python scripts/train.py --source ieee --n-samples 200000
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import mlflow
import numpy as np
import typer
from loguru import logger

from src.data.loader import FEATURE_COLS, LABEL_COL, load_dataset, split_dataset
from src.data.preprocessor import FraudPreprocessor
from src.models.evaluator import (
    evaluate_latency,
    full_report,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_score_distribution,
)
from src.models.fraud_detector import FraudEnsemble
from src.models.threshold_optimizer import CostSensitiveThresholdOptimizer
from src.monitoring.drift_detector import DriftDetector

ARTIFACT_DIR = ROOT / "artifacts" / "models"
MLFLOW_URI = "sqlite:///mlflow.db"
EXPERIMENT = "fraud-drift-pipeline"

app = typer.Typer(add_completion=False)


@app.command()
def train(
    source: str = typer.Option("synthetic"),
    n_samples: int = typer.Option(200_000),
    fraud_rate: float = typer.Option(0.005),
    fn_cost: float = typer.Option(90.0, help="Cost per missed fraud ($)"),
    fp_cost: float = typer.Option(25.0, help="Cost per false alarm ($)"),
    recall_target: float = typer.Option(0.80, help="Fixed recall target for P@R metric"),
    random_state: int = typer.Option(42),
):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info(f"Loading dataset: source={source}")
    df = load_dataset(source=source, n_samples=n_samples, fraud_rate=fraud_rate, random_state=random_state)
    train_df, val_df, test_df = split_dataset(df, random_state=random_state)

    # ── Preprocessing ─────────────────────────────────────────────────────────
    prep = FraudPreprocessor()
    X_train = prep.fit_transform(train_df)
    X_val   = prep.transform(val_df)
    X_test  = prep.transform(test_df)

    y_train = train_df[LABEL_COL].values.astype(np.int8)
    y_val   = val_df[LABEL_COL].values.astype(np.int8)
    y_test  = test_df[LABEL_COL].values.astype(np.int8)

    with mlflow.start_run(run_name="fraud_ensemble"):
        mlflow.log_params({
            "source": source,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "fraud_rate_train": round(y_train.mean(), 5),
            "fn_cost": fn_cost,
            "fp_cost": fp_cost,
            "recall_target": recall_target,
        })

        # ── Train ─────────────────────────────────────────────────────────────
        ensemble = FraudEnsemble(lgbm_weight=0.80)
        ensemble.fit(X_train, y_train, X_val, y_val)

        # ── Threshold optimisation (on val set) ───────────────────────────────
        val_scores = ensemble.predict_proba(X_val)
        optimizer = CostSensitiveThresholdOptimizer(
            fn_cost=fn_cost, fp_cost=fp_cost, recall_target=recall_target
        )
        optimizer.fit(y_val, val_scores)
        threshold = optimizer.optimal_threshold_

        # ── Test evaluation ───────────────────────────────────────────────────
        test_scores = ensemble.predict_proba(X_test)
        report = full_report(y_test, test_scores, optimal_threshold=threshold, recall_target=recall_target)
        pr_result = optimizer.precision_at_fixed_recall(y_test, test_scores, recall_target=recall_target)
        thr_result = optimizer.get_result(y_test, test_scores)

        # ── Latency ───────────────────────────────────────────────────────────
        latency = evaluate_latency(ensemble, X_test)

        # ── Log metrics ───────────────────────────────────────────────────────
        all_metrics = {
            **report,
            **{f"latency_{k}": v for k, v in latency.items() if k != "sla_met"},
            "sla_met": int(latency["sla_met"]),
            "precision_at_recall80": pr_result["precision_at_recall"],
            "optimal_threshold": threshold,
            "cost_saving_vs_default_usd": thr_result.cost_vs_default_saving_usd,
        }
        mlflow.log_metrics({k: float(v) for k, v in all_metrics.items() if isinstance(v, (int, float))})

        # ── Plots ─────────────────────────────────────────────────────────────
        y_pred = (test_scores >= threshold).astype(int)
        plots = [
            plot_confusion_matrix(y_test, y_pred, threshold),
            plot_score_distribution(y_test, test_scores, threshold),
            optimizer.plot_cost_curve(),
            optimizer.plot_precision_recall_curve(y_test, test_scores),
            plot_feature_importance(ensemble.feature_importance(FEATURE_COLS)),
        ]
        for p in plots:
            mlflow.log_artifact(str(p))

        # ── Save artefacts ────────────────────────────────────────────────────
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        prep.save()
        ensemble.save()
        np.save(ARTIFACT_DIR / "optimal_threshold.npy", np.array(threshold))

        # ── Fit & save DriftDetector on training reference ─────────────────────
        drift_det = DriftDetector(drift_threshold=0.05, alert_share_pct=0.20)
        drift_det.fit(train_df)
        joblib.dump(drift_det, ARTIFACT_DIR / "drift_detector.joblib")
        logger.info("DriftDetector saved.")

        # ── Print summary ─────────────────────────────────────────────────────
        print("\n══ TRAINING COMPLETE ══════════════════════════════")
        print(f"  Optimal threshold  : {threshold:.4f}")
        print(f"  Precision@Recall80 : {pr_result['precision_at_recall']:.4f}")
        print(f"  ROC-AUC            : {report['roc_auc']:.4f}")
        print(f"  PR-AUC             : {report['pr_auc']:.4f}")
        print(f"  P99 latency        : {latency['p99_ms']}ms  {'✓' if latency['sla_met'] else '✗ SLA BREACH'}")
        print(f"  Cost saving vs τ=0.5: ${thr_result.cost_vs_default_saving_usd:,.0f}")
        print("════════════════════════════════════════════════════\n")


if __name__ == "__main__":
    app()
