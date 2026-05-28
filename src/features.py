import pandas as pd
import numpy as np
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame,
                      is_training: bool = True) -> pd.DataFrame:
    """
    Engineer all features from raw BNPL dataset.

    Two model types:
      - Underwriting: features available at order time only
      - Early warning: includes installment 1 payment behaviour

    Args:
        df:          Raw dataframe
        is_training: Whether building training features

    Returns:
        Feature dataframe ready for model training or inference
    """
    logger.info(f"Engineering features for {len(df):,} records...")
    feat = df.copy()

    # ── Ensure all date columns are datetime ──────────────────────────────
    date_cols = [c for c in feat.columns if "date" in c]
    for col in date_cols:
        feat[col] = pd.to_datetime(feat[col], errors="coerce")

    # ── 1. Payment lateness features ─────────────────────────────────────
    # Days late = payment_date - due_date (positive = late, negative = early)
    for i in range(1, 5):
        due     = f"installment_{i}_due_date"
        paid    = f"installment_{i}_payment_date"
        late    = f"days_late_{i}"

        feat[late] = (
            feat[paid] - feat[due]
        ).dt.days.fillna(999)
        # 999 = not paid at all — strong default signal

    feat["max_days_late"]  = feat[["days_late_1","days_late_2",
                                    "days_late_3","days_late_4"]].max(axis=1)
    feat["avg_days_late"]  = feat[["days_late_1","days_late_2",
                                    "days_late_3","days_late_4"]].replace(999, np.nan).mean(axis=1).fillna(999)
    feat["any_late"]       = (
        feat[["days_late_1","days_late_2",
              "days_late_3","days_late_4"]]
        .apply(lambda x: ((x > 0) & (x < 999)).any(), axis=1)
    ).astype(int)

    feat["early_payment"]  = (
        feat[["days_late_1","days_late_2",
              "days_late_3","days_late_4"]]
        .apply(lambda x: (x < 0).any(), axis=1)
    ).astype(int)

    # ── 2. Missing payment features ───────────────────────────────────────
    # Early warning model — only use installment 1 missingness
    # Installments 2-4 are future data at prediction time
    feat["n_missing_payments"] = feat["installment_1_payment_date"].isnull().astype(int)
    feat["pct_paid"]           = (4 - feat["n_missing_payments"]) / 4
    feat["all_paid"]           = (feat["n_missing_payments"] == 0).astype(int)
    feat["none_paid"]          = (feat["n_missing_payments"] == 4).astype(int)

    # ── 3. Payment shortfall features ────────────────────────────────────
    # Shortfall = due - paid (positive = underpaid)
    for i in range(1, 5):
        due_amt  = f"installment_{i}_due_amount"
        paid_amt = f"installment_{i}_payment_amount"
        short    = f"shortfall_{i}"
        feat[short] = (
            feat[due_amt] - feat[paid_amt].fillna(0)
        ).clip(lower=0)

    feat["total_shortfall"]    = feat[["shortfall_1","shortfall_2",
                                       "shortfall_3","shortfall_4"]].sum(axis=1)
    feat["pct_amount_paid"]    = (
        feat[[f"installment_{i}_payment_amount" for i in range(1,5)]]
        .fillna(0).sum(axis=1) /
        feat[[f"installment_{i}_due_amount" for i in range(1,5)]]
        .sum(axis=1)
    ).clip(0, 1)

    feat["has_shortfall"]      = (feat["total_shortfall"] > 0).astype(int)

    # ── 4. Refund features ────────────────────────────────────────────────
    refund_cols = [f"installment_{i}_total_refund_amount" for i in range(1,5)]
    feat["total_refund"]       = feat[refund_cols].sum(axis=1)
    feat["has_refund"]         = (feat["total_refund"] > 0).astype(int)
    feat["n_refunds"]          = (feat[refund_cols] > 0).sum(axis=1)

    # ── 5. Order features ─────────────────────────────────────────────────
    feat["order_type_enc"]     = (feat["order_type"] == "extension").astype(int)
    feat["log_order_amount"]   = np.log1p(feat["order_amount"])

    # Order amount buckets
    feat["amount_bucket"] = pd.cut(
        feat["order_amount"],
        bins=[0, 50, 100, 200, 500, 99999],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    # ── 6. Customer history features ─────────────────────────────────────
    if is_training:
        # Sort by order date to avoid leakage
        feat = feat.sort_values("order_date")

        # Customer order count (how many orders has this customer had)
        feat["customer_order_count"] = feat.groupby(
            "customer_id"
        ).cumcount()

        # Customer prior default rate (expanding mean — no leakage)
        feat["customer_prior_default_rate"] = (
            feat.groupby("customer_id")["target"]
            .transform(lambda x: x.shift(1).expanding().mean())
            .fillna(-1)   # -1 = no prior orders
        )
    else:
        # At inference time these come from a feature store or lookup table
        feat["customer_order_count"]        = 0
        feat["customer_prior_default_rate"] = -1

    # ── 7. Merchant risk features ─────────────────────────────────────────
    if is_training:
        merchant_stats = (
            feat.groupby("merchant_id")["target"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "merchant_default_rate",
                              "count": "merchant_order_count"})
        )
        feat = feat.merge(merchant_stats, on="merchant_id", how="left")
    else:
        feat["merchant_default_rate"]  = 0.044  # Global default rate
        feat["merchant_order_count"]   = 0

    # ── 8. Timing features ────────────────────────────────────────────────
    feat["order_day_of_week"]  = feat["order_date"].dt.dayofweek
    feat["order_month"]        = feat["order_date"].dt.month

    # Days between installments (consistency signal)
    feat["days_gap_1_2"] = (
        feat["installment_2_due_date"] -
        feat["installment_1_due_date"]
    ).dt.days.fillna(14)

    logger.info(f"Feature engineering complete. "
                f"Features created: {len(get_feature_cols())}")
    return feat


def get_feature_cols() -> list:
    """
    Features available AFTER installment 1 is due.
    Early warning model — no leakage from future installments.
    """
    return [
        # Installment 1 behaviour only — known after first due date
        "days_late_1",          # Was installment 1 late?
        "shortfall_1",          # Was installment 1 underpaid?
        "n_missing_payments",   # How many total are missing so far (just inst 1 context)

        # Order characteristics — known at order time
        "order_type_enc",       # checkout vs extension
        "log_order_amount",     # order size
        "amount_bucket",        # order size bucket

        # Customer history — known at order time
        "customer_order_count",         # how many orders has this customer had
        "customer_prior_default_rate",  # have they defaulted before

        # Merchant risk — known at order time
        "merchant_default_rate",        # how risky is this merchant
        "merchant_order_count",         # merchant volume

        # Timing — known at order time
        "order_day_of_week",
        "order_month",
        "days_gap_1_2",         # installment schedule spacing

        # Refund from installment 1 only
        "has_refund",
        "total_refund",
    ]

def split_data(df: pd.DataFrame,
               test_size: float = 0.2) -> Tuple[pd.DataFrame,
                                                  pd.DataFrame]:
    """
    Time-based train/test split.
    Earlier orders = train. Later orders = test.
    Prevents data leakage from future to past.
    """
    df_sorted = df.sort_values("order_date")
    n_train   = int(len(df_sorted) * (1 - test_size))
    train     = df_sorted.iloc[:n_train].copy()
    test      = df_sorted.iloc[n_train:].copy()

    logger.info(f"Train: {len(train):,} rows "
                f"({train.target.mean()*100:.1f}% default)")
    logger.info(f"Test:  {len(test):,} rows "
                f"({test.target.mean()*100:.1f}% default)")
    return train, test