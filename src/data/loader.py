"""
src/data/loader.py

Two data sources:
  1. IEEE-CIS Fraud Detection (Kaggle) — if the user downloads it manually.
  2. Synthetic transaction simulator — realistic fraud patterns, no download needed.

Synthetic design
────────────────
• ~200k transactions, fraud rate ≈ 0.5% (mirrors real-world imbalance).
• Fraud transactions are generated from a distinct distribution:
    - unusually high amounts or micro-transactions (card testing)
    - international flag more common
    - high velocity (many transactions in last hour)
    - new accounts / recently issued cards
    - night-time transactions
• Provides `true_fraud_prob` for threshold calibration tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

FEATURE_COLS = [
    "transaction_amount",
    "hour_of_day",
    "day_of_week",
    "is_international",
    "merchant_category_encoded",
    "card_type_encoded",
    "distance_from_home_km",
    "time_since_last_txn_min",
    "n_txn_last_1h",
    "n_txn_last_24h",
    "n_txn_last_7d",
    "avg_amount_last_30d",
    "std_amount_last_30d",
    "amount_vs_avg_ratio",
    "card_age_days",
    "account_age_days",
    "is_weekend",
    "is_night",
    "velocity_score",
    "amount_log",
]

LABEL_COL = "is_fraud"
TRANSACTION_ID_COL = "transaction_id"

MERCHANT_CATEGORIES = [
    "grocery", "fuel", "restaurant", "travel", "electronics",
    "clothing", "healthcare", "entertainment", "online_retail", "atm",
]
CARD_TYPES = ["credit", "debit", "prepaid"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def generate_synthetic(
    n_samples: int = 200_000,
    fraud_rate: float = 0.005,
    random_state: int = 42,
    drift: bool = False,
) -> pd.DataFrame:
    """
    Generate realistic synthetic transaction data with heterogeneous fraud patterns.

    Parameters
    ----------
    drift : If True, shift feature distributions (simulates production drift).
            Used by the drift monitor tests.
    """
    rng = np.random.default_rng(random_state)
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    def _block(n: int, is_fraud: bool) -> dict:
        # Drift shifts the distribution if requested
        drift_shift = 0.5 if (drift and not is_fraud) else 0.0

        amount = (
            rng.choice(
                [rng.exponential(15, n), rng.lognormal(5, 1.5, n)],
                p=[0.3, 0.7],
            )
            if is_fraud
            else rng.lognormal(3.8 + drift_shift, 1.0, n)
        ).clip(0.5, 15_000)

        hour = (
            rng.choice(np.arange(24), n, p=_night_heavy_dist(rng))
            if is_fraud
            else rng.choice(np.arange(24), n)
        )
        day = rng.integers(0, 7, n)
        is_intl = rng.binomial(1, 0.35 if is_fraud else 0.08, n)
        merchant_cat = rng.integers(0, len(MERCHANT_CATEGORIES), n)
        card_type = rng.choice([0, 1, 2], n, p=[0.4, 0.5, 0.1] if not is_fraud else [0.2, 0.3, 0.5])
        dist_home = rng.exponential(200 if is_fraud else 15, n).clip(0, 5000)
        time_since = rng.exponential(5 if is_fraud else 300, n).clip(0.1, 10_000)
        n_1h = rng.integers(1, 15 if is_fraud else 4, n)
        n_24h = rng.integers(1, 40 if is_fraud else 10, n)
        n_7d = rng.integers(1, 100, n)
        avg_30d = rng.lognormal(3.8 + drift_shift, 0.8, n).clip(1, 5000)
        std_30d = rng.lognormal(2.5, 0.8, n).clip(0.1, 2000)
        card_age = rng.integers(1, 2000, n)
        acct_age = rng.integers(1 if is_fraud else 30, 3650, n)
        velocity_score = (n_1h * 3 + n_24h / 5).clip(0, 100)

        return {
            "transaction_amount": amount,
            "hour_of_day": hour,
            "day_of_week": day,
            "is_international": is_intl,
            "merchant_category_encoded": merchant_cat,
            "card_type_encoded": card_type,
            "distance_from_home_km": dist_home,
            "time_since_last_txn_min": time_since,
            "n_txn_last_1h": n_1h,
            "n_txn_last_24h": n_24h,
            "n_txn_last_7d": n_7d,
            "avg_amount_last_30d": avg_30d,
            "std_amount_last_30d": std_30d,
            "amount_vs_avg_ratio": amount / (avg_30d + 1e-6),
            "card_age_days": card_age,
            "account_age_days": acct_age,
            "is_weekend": (day >= 5).astype(int),
            "is_night": ((hour < 6) | (hour >= 23)).astype(int),
            "velocity_score": velocity_score,
            "amount_log": np.log1p(amount),
        }

    legit = _block(n_legit, is_fraud=False)
    fraud = _block(n_fraud, is_fraud=True)

    legit_df = pd.DataFrame(legit)
    legit_df[LABEL_COL] = 0
    fraud_df = pd.DataFrame(fraud)
    fraud_df[LABEL_COL] = 1

    df = pd.concat([legit_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df[TRANSACTION_ID_COL] = [f"TXN-{i:08d}" for i in range(len(df))]

    # Feature-type casts
    int_cols = [
        "hour_of_day", "day_of_week", "is_international",
        "merchant_category_encoded", "card_type_encoded",
        "n_txn_last_1h", "n_txn_last_24h", "n_txn_last_7d",
        "card_age_days", "account_age_days",
        "is_weekend", "is_night",
    ]
    for c in int_cols:
        df[c] = df[c].astype(np.int32)

    logger.info(
        f"Synthetic transactions: {len(df):,} | "
        f"fraud_rate={df[LABEL_COL].mean()*100:.2f}% | "
        f"drift={drift}"
    )
    return df


def _night_heavy_dist(rng: np.random.Generator) -> np.ndarray:
    """Probability distribution over 24 hours — heavier at night for fraud."""
    p = np.ones(24)
    p[0:6] *= 3.0   # 0–5 am heavier
    p[22:24] *= 2.0
    return p / p.sum()


def load_dataset(
    source: str = "synthetic",
    n_samples: int = 200_000,
    fraud_rate: float = 0.005,
    random_state: int = 42,
) -> pd.DataFrame:
    if source == "synthetic":
        return generate_synthetic(n_samples=n_samples, fraud_rate=fraud_rate, random_state=random_state)
    elif source == "ieee":
        path = RAW_DIR / "ieee_fraud_train.parquet"
        if not path.exists():
            raise FileNotFoundError(
                "IEEE-CIS dataset not found. Download from Kaggle and save as "
                "data/raw/ieee_fraud_train.parquet"
            )
        df = pd.read_parquet(path)
        logger.info(f"IEEE-CIS loaded: {len(df):,} rows")
        return df
    else:
        raise ValueError(f"Unknown source '{source}'. Choose 'synthetic' or 'ieee'.")


def split_dataset(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split preserving fraud rate in each fold."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[LABEL_COL], random_state=random_state
    )
    val_rel = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_rel, stratify=train_val[LABEL_COL], random_state=random_state
    )
    logger.info(
        f"Split → train={len(train):,} "
        f"val={len(val):,} "
        f"test={len(test):,}"
    )
    return train, val, test
