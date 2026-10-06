"""
src/models/fraud_detector.py

Two-model fraud detection ensemble:

  1. LightGBM  — primary gradient-boosted classifier; fast inference (<5ms),
                 handles class imbalance via scale_pos_weight.
  2. Isolation Forest — unsupervised anomaly detector; catches novel fraud
                        patterns not seen during training.
  3. Ensemble  — weighted average of LGBM score + normalised IF score.

Each model exposes:
  .fit(X_train, y_train, X_val, y_val)
  .predict_proba(X)  → fraud probability scores [0, 1]
  .save() / .load()

LightGBM uses categorical_feature natively — no one-hot encoding needed.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "models"

_LGBM_PARAMS = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "max_depth": -1,
    "learning_rate": 0.05,
    "n_estimators": 600,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM classifier
# ─────────────────────────────────────────────────────────────────────────────

class LGBMFraudModel:
    """
    LightGBM with scale_pos_weight computed from class imbalance ratio.
    Early stopping on validation AUC.
    """

    def __init__(self, categorical_feature_indices: list[int] | None = None):
        self._model: lgb.LGBMClassifier | None = None
        self.categorical_feature_indices = categorical_feature_indices or []
        self.name = "lgbm_fraud"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "LGBMFraudModel":
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale_pos_weight = neg / pos
        logger.info(
            f"LightGBM fitting | n_train={len(X_train):,} | "
            f"fraud_rate={pos/len(y_train)*100:.2f}% | "
            f"scale_pos_weight={scale_pos_weight:.1f}"
        )

        cat_cols = self.categorical_feature_indices if self.categorical_feature_indices else "auto"

        self._model = lgb.LGBMClassifier(
            **_LGBM_PARAMS,
            scale_pos_weight=scale_pos_weight,
        )
        self._model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            categorical_feature=cat_cols,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )
        logger.info(
            f"LightGBM done | best_iteration={self._model.best_iteration_}"
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return self._model.predict_proba(X)[:, 1]

    def feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        imp = self._model.feature_importances_
        return dict(sorted(zip(feature_names, imp), key=lambda x: -x[1]))

    def save(self, path: Path | None = None) -> Path:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        path = path or ARTIFACT_DIR / f"{self.name}.joblib"
        joblib.dump(self, path)
        logger.info(f"{self.name} saved → {path}")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "LGBMFraudModel":
        path = path or ARTIFACT_DIR / "lgbm_fraud.joblib"
        return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# Isolation Forest anomaly detector
# ─────────────────────────────────────────────────────────────────────────────

class IsolationForestModel:
    """
    Unsupervised anomaly detector trained on legitimate transactions only.
    Scores are inverted and normalised to [0, 1] so high = anomalous.
    """

    def __init__(self, contamination: float = 0.005):
        self.contamination = contamination
        self._model: IsolationForest | None = None
        self._score_scaler = MinMaxScaler()
        self.name = "isolation_forest"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs,
    ) -> "IsolationForestModel":
        # Train on legitimate transactions only — no label leakage
        X_legit = X_train[y_train == 0]
        logger.info(
            f"IsolationForest fitting on {len(X_legit):,} legit transactions "
            f"| contamination={self.contamination}"
        )
        self._model = IsolationForest(
            n_estimators=200,
            max_samples="auto",
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(X_legit)

        # Fit score normaliser on all training data
        raw_scores = -self._model.score_samples(X_train)  # higher = more anomalous
        self._score_scaler.fit(raw_scores.reshape(-1, 1))
        logger.info("IsolationForest done.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        raw = -self._model.score_samples(X)
        normalised = self._score_scaler.transform(raw.reshape(-1, 1)).ravel()
        return normalised.clip(0, 1)

    def save(self, path: Path | None = None) -> Path:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        path = path or ARTIFACT_DIR / f"{self.name}.joblib"
        joblib.dump(self, path)
        logger.info(f"{self.name} saved → {path}")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "IsolationForestModel":
        path = path or ARTIFACT_DIR / "isolation_forest.joblib"
        return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble
# ─────────────────────────────────────────────────────────────────────────────

class FraudEnsemble:
    """
    Weighted average: score = α × lgbm_score + (1−α) × if_score.
    α = 0.80 gives LightGBM dominant weight; IF acts as a novelty detector.
    """

    def __init__(self, lgbm_weight: float = 0.80):
        self.lgbm_weight = lgbm_weight
        self.if_weight = 1.0 - lgbm_weight
        self.lgbm: LGBMFraudModel | None = None
        self.isoforest: IsolationForestModel | None = None
        self.name = "fraud_ensemble"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "FraudEnsemble":
        self.lgbm = LGBMFraudModel()
        self.lgbm.fit(X_train, y_train, X_val, y_val)

        self.isoforest = IsolationForestModel(contamination=y_train.mean())
        self.isoforest.fit(X_train, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        lgbm_scores = self.lgbm.predict_proba(X)
        if_scores = self.isoforest.predict_proba(X)
        return self.lgbm_weight * lgbm_scores + self.if_weight * if_scores

    def feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        return self.lgbm.feature_importance(feature_names)

    def save(self, path: Path | None = None) -> Path:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        path = path or ARTIFACT_DIR / f"{self.name}.joblib"
        joblib.dump(self, path)
        logger.info(f"Ensemble saved → {path}")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "FraudEnsemble":
        path = path or ARTIFACT_DIR / "fraud_ensemble.joblib"
        return joblib.load(path)
