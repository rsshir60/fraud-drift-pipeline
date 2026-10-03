"""
src/data/preprocessor.py

Preprocessing pipeline for the fraud detection model.

Design decisions
────────────────
• No target-leaking aggregations — all rolling stats are pre-computed from
  historical windows, not from the label.
• RobustScaler instead of StandardScaler — transaction amounts are heavy-tailed;
  robust scaling prevents extreme outliers from dominating tree-split scores.
• Categorical features (merchant_category, card_type) are already label-encoded
  in the synthetic generator; LightGBM treats them as categoricals natively.
• Pipeline serialised with joblib for identical transforms at inference time.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.data.loader import FEATURE_COLS, LABEL_COL

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "models"

# Features that should NOT be scaled (already bounded / binary / categorical)
_PASSTHROUGH_COLS = [
    "hour_of_day", "day_of_week", "is_international",
    "merchant_category_encoded", "card_type_encoded",
    "is_weekend", "is_night",
]

_SCALE_COLS = [c for c in FEATURE_COLS if c not in _PASSTHROUGH_COLS]

# LightGBM categorical feature indices (0-based in FEATURE_COLS order)
CATEGORICAL_FEATURE_INDICES = [
    FEATURE_COLS.index(c)
    for c in ["merchant_category_encoded", "card_type_encoded", "is_international",
              "is_weekend", "is_night"]
]


class FraudPreprocessor:
    """
    Stateful transformer: fit on train, apply to val/test/live.

    Output is always a float32 numpy array — LightGBM and sklearn both accept it.
    Categorical indices are exposed so LightGBM can handle them natively.
    """

    def __init__(self):
        self.feature_names_: list[str] = FEATURE_COLS.copy()
        self.categorical_feature_indices_: list[int] = CATEGORICAL_FEATURE_INDICES.copy()
        self._scaler: RobustScaler | None = None
        self._scale_col_idx: list[int] = [
            FEATURE_COLS.index(c) for c in _SCALE_COLS
        ]

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_names_].values.astype(np.float32)
        scaler = RobustScaler()
        X[:, self._scale_col_idx] = scaler.fit_transform(
            X[:, self._scale_col_idx]
        )
        self._scaler = scaler
        logger.info(
            f"Preprocessor fitted on {len(df):,} rows | "
            f"{len(self.feature_names_)} features | "
            f"{len(_SCALE_COLS)} scaled cols"
        )
        return X

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self._scaler is None:
            raise RuntimeError("Call fit_transform() first.")
        X = df[self.feature_names_].values.astype(np.float32)
        X[:, self._scale_col_idx] = self._scaler.transform(X[:, self._scale_col_idx])
        return X

    def transform_dict(self, record: dict) -> np.ndarray:
        """Transform a single transaction dict — used in live inference."""
        row = {f: record.get(f, 0.0) for f in self.feature_names_}
        df = pd.DataFrame([row])
        return self.transform(df)

    def save(self, path: Path | None = None) -> Path:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        path = path or ARTIFACT_DIR / "preprocessor.joblib"
        joblib.dump(self, path)
        logger.info(f"Preprocessor saved → {path}")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "FraudPreprocessor":
        path = path or ARTIFACT_DIR / "preprocessor.joblib"
        obj = joblib.load(path)
        logger.info(f"Preprocessor loaded ← {path}")
        return obj
