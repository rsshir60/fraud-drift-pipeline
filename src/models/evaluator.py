"""
src/models/evaluator.py

Model evaluation metrics for fraud detection.

Metrics
───────
• Precision@FixedRecall  : Primary metric — what precision do we get
                           while catching 80% of fraud?
• P99 inference latency  : Must be < 50ms per the SLA.
• Cost-weighted score    : Total business loss at optimal threshold.
• Confusion matrix       : TP/FP/FN/TN breakdown.
• ROC-AUC, PR-AUC        : Standard auxiliary metrics.
• Feature importance     : LightGBM split-based importance.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

PLOT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "plots"


def evaluate_latency(
    model,
    X: np.ndarray,
    n_warmup: int = 50,
    n_timed: int = 500,
) -> dict[str, float]:
    """
    Measure P50/P95/P99 single-transaction inference latency.
    Warmup runs prevent cold-start JIT effects from skewing P99.

    SLA: P99 < 50ms.
    """
    # Warmup
    for i in range(n_warmup):
        model.predict_proba(X[i : i + 1])

    latencies_ms = []
    for i in range(n_timed):
        idx = i % len(X)
        t0 = time.perf_counter()
        model.predict_proba(X[idx : idx + 1])
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    result = {
        "p50_ms": round(float(np.percentile(latencies_ms, 50)), 3),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 3),
        "p99_ms": round(float(np.percentile(latencies_ms, 99)), 3),
        "mean_ms": round(float(np.mean(latencies_ms)), 3),
        "sla_met": float(np.percentile(latencies_ms, 99)) < 50.0,
    }
    logger.info(
        f"Latency — P50={result['p50_ms']}ms  "
        f"P95={result['p95_ms']}ms  "
        f"P99={result['p99_ms']}ms  "
        f"SLA(<50ms)={'✓' if result['sla_met'] else '✗'}"
    )
    return result


def full_report(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    optimal_threshold: float,
    recall_target: float = 0.80,
) -> dict:
    """Compute all classification metrics at the optimal cost threshold."""
    y_pred = (y_scores >= optimal_threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc  = average_precision_score(y_true, y_scores)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    result = {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "threshold": optimal_threshold,
        "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn),
        "fraud_flag_rate_pct": round((tp + fp) / len(y_pred) * 100, 2),
    }

    logger.info("\n" + "\n".join(f"  {k}: {v}" for k, v in result.items()))
    logger.info("\n" + classification_report(y_true, y_pred, target_names=["legit", "fraud"]))
    return result


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    save: bool = True,
) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    labels = ["Legit", "Fraud"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix  (τ={threshold:.3f})", fontsize=13)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cm[i,j] > cm.max() / 2 else "black", fontsize=13)
    fig.tight_layout()
    path = PLOT_DIR / "confusion_matrix.png"
    if save:
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_importance(
    importance_dict: dict[str, float],
    top_n: int = 15,
    save: bool = True,
) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    items = list(importance_dict.items())[:top_n]
    names = [i[0] for i in reversed(items)]
    vals  = [i[1] for i in reversed(items)]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(names, vals, color="#2196F3", alpha=0.85)
    ax.set_xlabel("LightGBM feature importance (split count)", fontsize=12)
    ax.set_title("Top Feature Importances — Fraud Detector", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    path = PLOT_DIR / "feature_importance.png"
    if save:
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_score_distribution(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
    save: bool = True,
) -> Path:
    """Fraud vs legitimate score distribution — shows model separation quality."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 80)
    ax.hist(y_scores[y_true == 0], bins=bins, alpha=0.6, color="#2196F3",
            label="Legitimate", density=True)
    ax.hist(y_scores[y_true == 1], bins=bins, alpha=0.7, color="#E91E63",
            label="Fraud", density=True)
    ax.axvline(threshold, color="red", linestyle="--", lw=2,
               label=f"Threshold = {threshold:.3f}")
    ax.set_xlabel("Fraud probability score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Score Distribution — Fraud vs Legitimate", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = PLOT_DIR / "score_distribution.png"
    if save:
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
