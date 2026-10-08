"""
src/models/threshold_optimizer.py

Asymmetric cost-sensitive decision threshold optimisation.

Why not F1?
───────────
F1 treats FP and FN as equally costly — they are not in fraud:
  • False Negative (missed fraud): issuer absorbs chargeback + investigation fee ≈ $90
  • False Positive (blocked legit): lost sale revenue + potential customer churn ≈ $25

Optimal threshold = argmin_t { FN_cost × FN(t) + FP_cost × FP(t) }

Also exposes:
  precision_at_fixed_recall : find threshold that hits a recall target,
                              then report precision at that point.
  threshold_sweep_report    : full table of (threshold, precision, recall,
                              f1, cost) for a range of thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import precision_recall_curve, roc_auc_score

PLOT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "plots"


@dataclass
class ThresholdResult:
    optimal_threshold: float
    fn_cost_usd: float
    fp_cost_usd: float
    total_cost_usd: float
    n_fn: int
    n_fp: int
    n_tp: int
    n_tn: int
    precision: float
    recall: float
    f1: float
    cost_vs_default_saving_usd: float  # saving vs default 0.5 threshold


class CostSensitiveThresholdOptimizer:
    """
    Finds the decision threshold that minimises total business cost.

    Parameters
    ----------
    fn_cost : Cost per missed fraud (chargeback + investigation)
    fp_cost : Cost per blocked legitimate transaction (lost sale)
    """

    def __init__(
        self,
        fn_cost: float = 90.0,
        fp_cost: float = 25.0,
        recall_target: float = 0.80,
    ):
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        self.recall_target = recall_target
        self.optimal_threshold_: float | None = None
        self.sweep_df_: pd.DataFrame | None = None

    def fit(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
    ) -> "CostSensitiveThresholdOptimizer":
        """Sweep thresholds [0.01, 0.99] and find the cost minimum."""
        thresholds = np.linspace(0.01, 0.99, 500)
        rows = []

        for t in thresholds:
            y_pred = (y_scores >= t).astype(int)
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            tn = int(((y_pred == 0) & (y_true == 0)).sum())

            cost = self.fn_cost * fn + self.fp_cost * fp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            rows.append(
                {
                    "threshold": round(t, 4),
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "total_cost_usd": cost,
                    "fn_cost_usd": self.fn_cost * fn,
                    "fp_cost_usd": self.fp_cost * fp,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )

        self.sweep_df_ = pd.DataFrame(rows)
        best_row = self.sweep_df_.loc[self.sweep_df_["total_cost_usd"].idxmin()]
        self.optimal_threshold_ = float(best_row["threshold"])

        # Cost at default 0.5 threshold
        default_cost = self.sweep_df_.loc[
            (self.sweep_df_["threshold"] - 0.5).abs().idxmin(),
            "total_cost_usd",
        ]

        logger.info(
            f"Optimal threshold: {self.optimal_threshold_:.4f} | "
            f"cost=${best_row['total_cost_usd']:,.0f} | "
            f"precision={best_row['precision']:.3f} | "
            f"recall={best_row['recall']:.3f} | "
            f"saving_vs_0.5=${default_cost - best_row['total_cost_usd']:,.0f}"
        )
        return self

    def precision_at_fixed_recall(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        recall_target: float | None = None,
    ) -> dict[str, float]:
        """
        Find the highest threshold that still achieves at least `recall_target`.
        Reports precision at that threshold — the key production metric.
        """
        target = recall_target or self.recall_target
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

        # Walk thresholds from high → low; stop when recall >= target
        # precision_recall_curve returns arrays sorted by decreasing threshold
        valid = recalls[:-1] >= target
        if not valid.any():
            logger.warning(
                f"Recall target {target} unreachable. "
                f"Returning lowest possible threshold results."
            )
            idx = len(thresholds) - 1
        else:
            idx = np.argmax(valid)  # first idx (highest threshold) that hits target

        result = {
            "recall_target": target,
            "achieved_recall": round(float(recalls[idx]), 4),
            "precision_at_recall": round(float(precisions[idx]), 4),
            "threshold": round(float(thresholds[idx]), 4),
            "roc_auc": round(float(roc_auc_score(y_true, y_scores)), 4),
        }
        logger.info(
            f"Precision@Recall{target}: {result['precision_at_recall']:.4f} "
            f"(threshold={result['threshold']:.4f})"
        )
        return result

    def get_result(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
    ) -> ThresholdResult:
        """Full ThresholdResult at the optimal threshold."""
        if self.optimal_threshold_ is None:
            raise RuntimeError("Call fit() first.")
        t = self.optimal_threshold_
        best = self.sweep_df_.loc[
            (self.sweep_df_["threshold"] - t).abs().idxmin()
        ]
        default_cost = self.sweep_df_.loc[
            (self.sweep_df_["threshold"] - 0.5).abs().idxmin(),
            "total_cost_usd",
        ]
        return ThresholdResult(
            optimal_threshold=t,
            fn_cost_usd=float(best["fn_cost_usd"]),
            fp_cost_usd=float(best["fp_cost_usd"]),
            total_cost_usd=float(best["total_cost_usd"]),
            n_fn=int(best["fn"]),
            n_fp=int(best["fp"]),
            n_tp=int(best["tp"]),
            n_tn=int(best["tn"]),
            precision=float(best["precision"]),
            recall=float(best["recall"]),
            f1=float(best["f1"]),
            cost_vs_default_saving_usd=float(default_cost - best["total_cost_usd"]),
        )

    def plot_cost_curve(self, save: bool = True) -> Path:
        """Plot total cost vs threshold — marks optimal point."""
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        if self.sweep_df_ is None:
            raise RuntimeError("Call fit() first.")

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()

        ax1.plot(
            self.sweep_df_["threshold"],
            self.sweep_df_["total_cost_usd"] / 1000,
            color="#E91E63", lw=2, label="Total cost ($k)",
        )
        ax1.plot(
            self.sweep_df_["threshold"],
            self.sweep_df_["fn_cost_usd"] / 1000,
            color="#FF9800", lw=1.5, linestyle="--", label="FN cost ($k)",
        )
        ax1.plot(
            self.sweep_df_["threshold"],
            self.sweep_df_["fp_cost_usd"] / 1000,
            color="#2196F3", lw=1.5, linestyle="--", label="FP cost ($k)",
        )
        ax1.axvline(self.optimal_threshold_, color="red", linestyle=":", lw=2, label=f"Optimal τ={self.optimal_threshold_:.3f}")

        ax2.plot(
            self.sweep_df_["threshold"],
            self.sweep_df_["precision"],
            color="#4CAF50", lw=1.5, alpha=0.6, label="Precision",
        )
        ax2.plot(
            self.sweep_df_["threshold"],
            self.sweep_df_["recall"],
            color="#9C27B0", lw=1.5, alpha=0.6, label="Recall",
        )

        ax1.set_xlabel("Decision threshold", fontsize=12)
        ax1.set_ylabel("Business cost ($k)", fontsize=12, color="#E91E63")
        ax2.set_ylabel("Precision / Recall", fontsize=12, color="#4CAF50")
        ax1.set_title("Cost-Sensitive Threshold Optimisation", fontsize=13)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
        ax1.grid(alpha=0.3)
        fig.tight_layout()

        path = PLOT_DIR / "cost_threshold_curve.png"
        if save:
            fig.savefig(path, dpi=150)
            logger.info(f"Cost curve saved → {path}")
        plt.close(fig)
        return path

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        save: bool = True,
    ) -> Path:
        """PR curve with recall target annotated."""
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        pr_result = self.precision_at_fixed_recall(y_true, y_scores)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recalls, precisions, color="#2196F3", lw=2, label="PR curve")
        ax.axvline(self.recall_target, color="#E91E63", linestyle="--", lw=1.5,
                   label=f"Recall target = {self.recall_target}")
        ax.scatter(
            pr_result["achieved_recall"],
            pr_result["precision_at_recall"],
            s=120, color="red", zorder=5,
            label=f"Precision@Recall={pr_result['precision_at_recall']:.3f}",
        )
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(f"Precision-Recall Curve  |  AUC-ROC={pr_result['roc_auc']:.4f}", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()

        path = PLOT_DIR / "precision_recall_curve.png"
        if save:
            fig.savefig(path, dpi=150)
            logger.info(f"PR curve saved → {path}")
        plt.close(fig)
        return path
