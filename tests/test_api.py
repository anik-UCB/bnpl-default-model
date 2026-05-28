import pytest
from fastapi.testclient import TestClient
import sys
sys.path.append("src")
sys.path.append("api")

from main import app

client = TestClient(app)

# ── Test data ─────────────────────────────────────────────────────────────
GOOD_CUSTOMER = {
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
}

RISKY_CUSTOMER = {
    "days_late_1": 999,
    "shortfall_1": 25.0,
    "n_missing_payments": 1,
    "order_type_enc": 1,
    "log_order_amount": 5.5,
    "amount_bucket": 3,
    "customer_order_count": 2,
    "customer_prior_default_rate": 0.5,
    "merchant_default_rate": 0.09,
    "merchant_order_count": 50,
    "order_day_of_week": 4,
    "order_month": 12,
    "days_gap_1_2": 14.0,
    "has_refund": 1,
    "total_refund": 25.0
}


# ── Tests ──────────────────────────────────────────────────────────────────
def test_health_check():
    """API should return healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] == True


def test_root_endpoint():
    """Root endpoint should return API info."""
    response = client.get("/")
    assert response.status_code == 200
    assert "BNPL" in response.json()["name"]


def test_predict_good_customer():
    """Good customer should return LOW risk."""
    response = client.post("/predict", json=GOOD_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_tier"] == "LOW"
    assert data["default_probability"] < 0.20
    assert "default_probability" in data
    assert "risk_score" in data
    assert "top_risk_factors" in data
    assert "recommendation" in data


def test_predict_risky_customer():
    """Risky customer should return HIGH risk."""
    response = client.post("/predict", json=RISKY_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_tier"] == "HIGH"
    assert data["default_probability"] > 0.50


def test_predict_returns_required_fields():
    """Prediction response must contain all required fields."""
    response = client.post("/predict", json=GOOD_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    required_fields = [
        "default_probability", "risk_tier", "risk_score",
        "top_risk_factors", "recommendation",
        "model_version", "timestamp"
    ]
    for field in required_fields:
        assert field in data, f"Missing field: {field}"


def test_predict_probability_range():
    """Default probability must be between 0 and 1."""
    response = client.post("/predict", json=GOOD_CUSTOMER)
    assert response.status_code == 200
    prob = response.json()["default_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_risk_score_range():
    """Risk score must be between 0 and 1000."""
    response = client.post("/predict", json=GOOD_CUSTOMER)
    assert response.status_code == 200
    score = response.json()["risk_score"]
    assert 0 <= score <= 1000


def test_predict_invalid_input():
    """Wrong data types should return 422."""
    response = client.post("/predict", json={
        "days_late_1": "not_a_number",
        "shortfall_1": "invalid"
    })
    assert response.status_code == 422


def test_batch_predict():
    """Batch endpoint should return predictions for all inputs."""
    response = client.post(
        "/predict/batch",
        json=[GOOD_CUSTOMER, RISKY_CUSTOMER]
    )
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 2
    assert results[0]["risk_tier"] == "LOW"
    assert results[1]["risk_tier"] == "HIGH"


def test_latency():
    """API should respond within 500ms."""
    import time
    start    = time.time()
    response = client.post("/predict", json=GOOD_CUSTOMER)
    elapsed  = (time.time() - start) * 1000
    assert response.status_code == 200
    assert elapsed < 500, f"Too slow: {elapsed:.0f}ms"