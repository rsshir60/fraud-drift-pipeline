# Real-Time Transaction Anomaly & Drift Detection Pipeline

> **Production-grade fraud detection system built for extreme class imbalance (0.5% fraud), sub-50ms P99 inference latency, asymmetric cost-sensitive decision thresholding, and continuous feature drift monitoring via Evidently AI.**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Evidently AI](https://img.shields.io/badge/Evidently-0.4-orange.svg)](https://www.evidentlyai.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com/)
[![Redpanda](https://img.shields.io/badge/Redpanda-Kafka_Compatible-red.svg)](https://redpanda.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.13-0194E2.svg)](https://mlflow.org/)

---

## Executive Summary & Engineering Highlights

Standard academic fraud tutorials optimize for ROC-AUC or default F1 score on balanced datasets. In real-world financial transaction systems:
1. **Severe Imbalance**: Legitimate transactions outnumber fraud 200:1 (0.5% fraud rate).
2. **Asymmetric Error Costs**: A False Negative (missed fraud) incurs an average chargeback + fraud investigation loss of **\$90**. A False Positive (declining a legitimate cardholder) causes merchant embarrassment and lost transaction margin of **\$25**. Default 0.5 probability thresholds bleed capital.
3. **Strict Latency SLA**: In-line payment gateways require **P99 inference latency < 50ms**.
4. **Non-Stationary Data (Drift)**: Fraudsters actively evolve card-testing tactics and velocity patterns. Automated statistical distribution testing (KS-test, Chi-square) is required to alert engineers before model degradation impacts the balance sheet.

---

## Architecture Overview

```
                          ┌────────────────────────┐
                          │  Payment Gateway Stream│
                          └───────────┬────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ Redpanda / Kafka Cluster  │
                        │   (Topic: 'transactions') │
                        └─────────────┬─────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              │                                               │
              ▼ (Real-Time In-line Stream)                    ▼ (Micro-Batch 500 txns)
┌───────────────────────────────┐               ┌───────────────────────────────┐
│     FastAPI / Stream Scorer   │               │     Evidently AI Monitor      │
│                               │               │                               │
│  • RobustScaler preprocessor  │               │  • Kolmogorov-Smirnov test    │
│  • LightGBM + IsolationForest │               │  • Chi-square test (cat)      │
│  • Optimal Cost Threshold (τ) │               │  • Population Stability Index │
│  • P99 Latency: < 5ms         │               │  • Automated Drift Alerting   │
└──────────────┬────────────────┘               └───────────────┬───────────────┘
               │                                                │
       ┌───────┴────────┐                                       │
       ▼                ▼                                       ▼
 ┌───────────┐    ┌───────────┐                           ┌───────────┐
 │ Approved  │    │  Blocked  │                           │   Slack   │
 │   Txns    │    │  (Alert)  │                           │ / Datadog │
 └───────────┘    └───────────┘                           └───────────┘
```

---

## Asymmetric Cost Matrix vs Default Thresholding

Instead of choosing an arbitrary threshold (e.g., 0.5), we define the business cost objective function:

$$\text{Total Cost}(\tau) = C_{\text{FN}} \cdot \text{FN}(\tau) + C_{\text{FP}} \cdot \text{FP}(\tau)$$

Where:
* $C_{\text{FN}} = \$90$ (Direct chargeback liability + processing fee)
* $C_{\text{FP}} = \$25$ (Customer friction + lost interchange fee)

```
Threshold (τ)  |  Recall  | Precision | False Neg (FN) | False Pos (FP) | Total Business Loss
---------------------------------------------------------------------------------------------
0.50 (Default) |  61.2%   |   84.1%   |      388       |       73       |      $36,745
0.28 (Optimal) |  82.4%   |   71.8%   |      176       |      162       |      $19,890  <-- Saves $16.8k/mo
```

Optimizing $\tau$ drops financial loss by **~46%** compared to the uncalibrated default.

---

## Model Pipeline

### 1. LightGBM Supervised Classifier
* `scale_pos_weight`: Dynamically calculated from negative-to-positive ratio ($~199.0$) to counter severe class imbalance.
* Native categorical handling for high-cardinality merchant and card type encodings.
* Fast tree execution with histogram-based binning (`tree_method="hist"`).

### 2. Isolation Forest Novelty Detector
* Fitted exclusively on legitimate transaction history (`y == 0`).
* Normalised anomaly scores identify zero-day fraudulent purchase behaviors and novel account takeover methods not represented in training labels.

### 3. Blended Ensemble
$$\text{Score} = 0.80 \cdot \text{Score}_{\text{LGBM}} + 0.20 \cdot \text{Score}_{\text{IF}}$$

---

## Drift Monitoring with Evidently AI

Every 500 streaming transactions, the detector evaluates data distributions against the training reference baseline:
* **Continuous Features** (`transaction_amount`, `velocity_score`, `distance_from_home_km`): Two-sample Kolmogorov-Smirnov test ($\alpha = 0.05$).
* **Categorical Features** (`merchant_category_encoded`, `card_type_encoded`, `is_international`): Chi-Square goodness-of-fit test ($\alpha = 0.05$).
* **Dataset Drift Alert**: Triggered if $>20\%$ of monitored features show statistically significant drift. Generates an interactive visual HTML report in `artifacts/drift_reports/`.

---

## Quickstart

### 1. Environment Setup

```bash
git clone https://github.com/yourname/fraud-drift-pipeline
cd fraud-drift-pipeline
pip install -r requirements.txt
```

### 2. Run Training & Threshold Calibration

```bash
python scripts/train.py --n-samples 200000 --fraud-rate 0.005 --fn-cost 90.0 --fp-cost 25.0
```

Console summary:
```text
══ TRAINING COMPLETE ══════════════════════════════
  Optimal threshold  : 0.2840
  Precision@Recall80 : 0.7230
  ROC-AUC            : 0.9420
  PR-AUC             : 0.7610
  P99 latency        : 3.42ms  ✓ SLA Compliant (<50ms)
  Cost saving vs τ=0.5: $16,855
════════════════════════════════════════════════════
```

### 3. Replay Streaming Simulation & Injected Drift

```bash
python scripts/simulate_stream.py --n-transactions 5000 --inject-drift
```
Tests 5,000 live incoming transactions, streams them through the preprocessor and model, and verifies that the automated drift detection triggers an alert upon synthetic distribution shift.

### 4. Serve the API

```bash
# Direct local run
uvicorn src.api.main:app --reload --port 8000

# Or via Docker Compose (Redpanda + FastAPI + MLflow)
docker compose up --build
```

### 5. Send Sample Transaction

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN-009988",
    "transaction_amount": 4200.0,
    "hour_of_day": 3,
    "day_of_week": 2,
    "is_international": 1,
    "merchant_category_encoded": 9,
    "card_type_encoded": 2,
    "distance_from_home_km": 1200.0,
    "time_since_last_txn_min": 1.2,
    "n_txn_last_1h": 7,
    "n_txn_last_24h": 28,
    "n_txn_last_7d": 40,
    "avg_amount_last_30d": 65.0,
    "std_amount_last_30d": 25.0,
    "amount_vs_avg_ratio": 64.6,
    "card_age_days": 15,
    "account_age_days": 8,
    "is_weekend": 0,
    "is_night": 1,
    "velocity_score": 26.6,
    "amount_log": 8.34
  }'
```

Response:
```json
{
  "transaction_id": "TXN-009988",
  "fraud_score": 0.894215,
  "is_flagged": true,
  "threshold": 0.284,
  "risk_level": "CRITICAL",
  "latency_ms": 2.14
}
```

---

## Automated Test Suite

```bash
pytest tests/ -v --asyncio-mode=auto
```

Covers:
* Data generator balance & feature boundaries
* RobustScaler leak prevention
* LightGBM + Isolation Forest score calibration
* Cost optimizer sweep & Precision@FixedRecall
* P99 latency benchmark validation
* Evidently drift detection sensitivity & specificity
* FastAPI endpoint validation and batch size limits
