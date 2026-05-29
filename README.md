# BNPL Default Prediction Model

An end-to-end machine learning system that predicts default risk on Buy Now Pay Later (BNPL) orders. The model is an **early-warning classifier**: it scores orders after installment 1 is due, using payment behaviour, customer history, and merchant risk signals.

Built with XGBoost, served via FastAPI, and deployed to production with Docker and GitHub Actions.

---

## Overview

| Component | Description |
|-----------|-------------|
| **Model** | XGBoost binary classifier with Optuna hyperparameter tuning and SMOTE for class imbalance |
| **Prediction timing** | After installment 1 due date — no leakage from future installments |
| **API** | FastAPI REST service with single and batch prediction endpoints |
| **Monitoring** | Evidently drift reports and PSI checks on production inputs |
| **Deployment** | Docker container deployed to Railway via CI/CD |

### Risk tiers

| Tier | Default probability | Recommended action |
|------|---------------------|--------------------|
| **LOW** | &lt; 10% | Standard monitoring |
| **MEDIUM** | 10% – 30% | Monitor closely; send payment reminder |
| **HIGH** | ≥ 30% | Flag for manual review; consider collections outreach |

---

## Project structure

```
bnpl-default-model/
├── api/
│   └── main.py              # FastAPI prediction service
├── src/
│   ├── features.py          # Feature engineering pipeline
│   ├── train.py             # Training with Optuna + MLflow
│   └── evaluate.py          # Threshold analysis on holdout set
├── monitoring/
│   └── monitor.py           # Drift detection (Evidently + PSI)
├── tests/
│   └── test_api.py          # API integration tests
├── models/
│   ├── xgboost_model.pkl    # Trained model artifact
│   └── feature_cols.pkl     # Feature column order
├── data/
│   └── data_bnpl.xlsx       # Training dataset
├── Dockerfile
├── requirements.txt
└── .github/workflows/
    └── deploy.yml           # Test → Build → Deploy pipeline
```

---

## Quick start

### Prerequisites

- Python 3.11+
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

For training, also install:

```bash
pip install mlflow optuna scipy evidently
```

### Run the API locally

From the project root:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs).

### Run with Docker

```bash
docker build -t bnpl-model .
docker run -p 8000:8000 bnpl-model
```

Verify the service is healthy:

```bash
curl http://localhost:8000/health
```

---

## API reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | API metadata |
| `GET` | `/health` | Health check and model status |
| `POST` | `/predict` | Single-order default prediction |
| `POST` | `/predict/batch` | Batch predictions |

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "days_late_1": 0,
    "shortfall_1": 0.0,
    "n_missing_payments": 0,
    "order_type_enc": 0,
    "log_order_amount": 4.5,
    "amount_bucket": 2,
    "customer_order_count": 5,
    "customer_prior_default_rate": 0.0,
    "merchant_default_rate": 0.03,
    "merchant_order_count": 200,
    "order_day_of_week": 1,
    "order_month": 6,
    "days_gap_1_2": 14.0,
    "has_refund": 0,
    "total_refund": 0.0
  }'
```

### Example response

```json
{
  "default_probability": 0.0421,
  "risk_tier": "LOW",
  "risk_score": 958,
  "top_risk_factors": ["No significant risk factors detected"],
  "recommendation": "No action required. Standard monitoring.",
  "model_version": "1.0.0",
  "timestamp": "2026-05-28T12:00:00.000000"
}
```

### Input features

All features are available after installment 1 is due:

| Feature | Type | Description |
|---------|------|-------------|
| `days_late_1` | float | Days late on installment 1 (`0` = on time, `999` = not paid) |
| `shortfall_1` | float | Unpaid amount on installment 1 |
| `n_missing_payments` | int | Whether installment 1 was missed (`0` or `1`) |
| `order_type_enc` | int | `0` = checkout, `1` = extension |
| `log_order_amount` | float | Log of order amount |
| `amount_bucket` | int | Order size bucket (`0`–`4`) |
| `customer_order_count` | int | Number of prior orders |
| `customer_prior_default_rate` | float | Prior default rate (`-1` = new customer) |
| `merchant_default_rate` | float | Merchant historical default rate |
| `merchant_order_count` | int | Total orders at merchant |
| `order_day_of_week` | int | Day order was placed (`0` = Monday) |
| `order_month` | int | Month order was placed |
| `days_gap_1_2` | float | Days between installment 1 and 2 due dates |
| `has_refund` | int | Whether a refund was issued (`0` or `1`) |
| `total_refund` | float | Total refund amount |

Predictions are logged to `predictions.jsonl` for downstream monitoring.

---

## Model training

### Pipeline

1. Load raw BNPL data from `data/data_bnpl.xlsx`
2. Engineer features with leakage-safe logic (installment 1 only)
3. Time-based train/test split (earlier orders → train, later → test)
4. Apply SMOTE to address class imbalance
5. Tune hyperparameters with Optuna (maximizing Gini)
6. Train final XGBoost model and log metrics to MLflow
7. Save model and feature list to `models/`

### Run training

```bash
cd src
python train.py
```

### Evaluate thresholds

```bash
cd src
python evaluate.py
```

Prints precision, recall, and flagged-rate across decision thresholds.

### Key metrics

The training pipeline tracks credit-risk metrics:

- **Gini** — `2 × AUC − 1`
- **KS statistic** — maximum separation between good and bad score distributions
- **AUC-ROC, precision, recall, F1**

---

## Monitoring

The monitoring module compares production inputs against the training reference distribution.

```bash
python monitoring/monitor.py
```

It produces:

- **PSI checks** on key features (Population Stability Index)
- **Evidently drift report** (HTML saved to `monitoring/reports/`)
- **JSON summary** with action recommendations

Drift triggers when more than 30% of features shift, or when critical payment features (`days_late_1`, `shortfall_1`) drift.

---

## CI/CD

GitHub Actions (`.github/workflows/deploy.yml`) runs on every push and pull request to `main`:

1. **Test** — install dependencies and run `pytest tests/ -v`
2. **Build** — build Docker image and verify `/health` responds
3. **Deploy** — on `main` only, deploy to Railway via `railway up`

Required GitHub secrets:

| Secret | Purpose |
|--------|---------|
| `RAILWAY_TOKEN` | Railway CLI authentication |
| `RAILWAY_PROJECT_ID` | Target Railway project |

---

## Testing

```bash
pytest tests/ -v
```

Tests cover health checks, single and batch prediction, risk tier logic, input validation, response schema, and latency (&lt; 500 ms).

---

## Tech stack

- **ML:** XGBoost, scikit-learn, imbalanced-learn (SMOTE), Optuna
- **Experiment tracking:** MLflow
- **API:** FastAPI, Uvicorn, Pydantic
- **Monitoring:** Evidently AI
- **Infrastructure:** Docker, GitHub Actions, Railway

---

## License

This project is provided for educational and portfolio purposes.
