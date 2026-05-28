from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Load model and feature list at startup ────────────────────────────────────
MODEL_PATH    = Path("models/xgboost_model.pkl")
FEATURES_PATH = Path("models/feature_cols.pkl")

try:
    model    = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    logger.info(f"Model loaded. Features: {len(features)}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model, features = None, None


# ── Request schema ────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    """
    Input features for BNPL default prediction.
    All fields are available after installment 1 is due.
    """
    # Installment 1 behaviour
    days_late_1:       float = Field(default=0,    description="Days late on installment 1 (0=on time, 999=not paid)")
    shortfall_1:       float = Field(default=0.0,  description="Amount unpaid on installment 1")
    n_missing_payments: int  = Field(default=0,    description="Was installment 1 missed? (0 or 1)")

    # Order characteristics
    order_type_enc:    int   = Field(default=0,    description="0=checkout, 1=extension")
    log_order_amount:  float = Field(default=4.5,  description="Log of order amount")
    amount_bucket:     int   = Field(default=1,    description="Order size bucket 0-4")

    # Customer history
    customer_order_count:        int   = Field(default=0,    description="Number of prior orders")
    customer_prior_default_rate: float = Field(default=-1.0, description="Prior default rate (-1=new customer)")

    # Merchant risk
    merchant_default_rate:  float = Field(default=0.044, description="Merchant historical default rate")
    merchant_order_count:   int   = Field(default=100,   description="Total orders at this merchant")

    # Timing
    order_day_of_week: int   = Field(default=1,    description="Day of week order placed (0=Mon)")
    order_month:       int   = Field(default=6,    description="Month order placed")
    days_gap_1_2:      float = Field(default=14.0, description="Days between installment 1 and 2 due dates")

    # Refunds
    has_refund:   int   = Field(default=0,   description="Was any refund issued? (0 or 1)")
    total_refund: float = Field(default=0.0, description="Total refund amount")


# ── Response schema ───────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    default_probability: float
    risk_tier:           str
    risk_score:          int
    top_risk_factors:    list
    recommendation:      str
    model_version:       str
    timestamp:           str


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "BNPL Default Prediction API",
    description = "Predicts probability of default on Buy Now Pay Later orders",
    version     = "1.0.0"
)


def get_risk_tier(prob: float) -> str:
    if prob >= 0.30:   return "HIGH"
    elif prob >= 0.10: return "MEDIUM"
    else:              return "LOW"


def get_recommendation(tier: str) -> str:
    return {
        "HIGH":   "Flag for manual review. Consider collections outreach.",
        "MEDIUM": "Monitor closely. Send payment reminder.",
        "LOW":    "No action required. Standard monitoring."
    }[tier]


def get_top_risk_factors(input_data: dict,
                          proba: float) -> list:
    """Return top risk factors in plain English."""
    factors = []

    if input_data["days_late_1"] == 999:
        factors.append("Installment 1 not paid at all")
    elif input_data["days_late_1"] > 7:
        factors.append(f"Installment 1 paid {input_data['days_late_1']:.0f} days late")

    if input_data["shortfall_1"] > 0:
        factors.append(f"Shortfall of ${input_data['shortfall_1']:.2f} on installment 1")

    if input_data["n_missing_payments"] == 1:
        factors.append("Installment 1 payment missing")

    if input_data["customer_prior_default_rate"] > 0.1:
        factors.append(f"Customer has prior default history "
                       f"({input_data['customer_prior_default_rate']*100:.0f}% rate)")

    if input_data["merchant_default_rate"] > 0.08:
        factors.append(f"High-risk merchant "
                       f"({input_data['merchant_default_rate']*100:.1f}% default rate)")

    if input_data["order_type_enc"] == 1:
        factors.append("Extension order — 2x higher default rate than checkout")

    if input_data["log_order_amount"] > 5.5:
        factors.append(f"High order amount "
                       f"(${np.expm1(input_data['log_order_amount']):.0f})")

    if not factors:
        factors.append("No significant risk factors detected")

    return factors[:3]   # Return top 3 only


@app.get("/health")
def health():
    return {
        "status":       "healthy",
        "model_loaded": model is not None,
        "n_features":   len(features) if features else 0,
        "timestamp":    datetime.utcnow().isoformat()
    }


@app.get("/")
def root():
    return {
        "name":        "BNPL Default Prediction API",
        "version":     "1.0.0",
        "docs":        "/docs",
        "health":      "/health",
        "predict":     "/predict"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )

    try:
        # Build feature dict in correct order
        input_dict = request.model_dump()
        input_df   = pd.DataFrame([input_dict])[features]

        # Predict
        proba     = float(model.predict_proba(input_df)[0][1])
        risk_tier = get_risk_tier(proba)

        # Risk score 0-1000 (like a credit score, inverted)
        risk_score = int((1 - proba) * 1000)

        # Log prediction
        logger.info(f"Prediction: prob={proba:.4f} tier={risk_tier}")

        # Log to JSONL for monitoring
        log_entry = {
            "timestamp":          datetime.utcnow().isoformat(),
            "input":              input_dict,
            "default_probability": proba,
            "risk_tier":          risk_tier,
        }
        with open("predictions.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return PredictionResponse(
            default_probability = round(proba, 4),
            risk_tier           = risk_tier,
            risk_score          = risk_score,
            top_risk_factors    = get_top_risk_factors(input_dict, proba),
            recommendation      = get_recommendation(risk_tier),
            model_version       = "1.0.0",
            timestamp           = datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(requests: list[PredictionRequest]):
    """Batch prediction endpoint for multiple orders."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        rows   = [r.model_dump() for r in requests]
        df     = pd.DataFrame(rows)[features]
        probas = model.predict_proba(df)[:, 1]

        return [
            {
                "index":               i,
                "default_probability": round(float(p), 4),
                "risk_tier":          get_risk_tier(float(p)),
                "risk_score":         int((1 - float(p)) * 1000),
            }
            for i, p in enumerate(probas)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))