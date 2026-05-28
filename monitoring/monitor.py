import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from datetime import datetime
import sys
sys.path.append("src")

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnDriftMetric,
)
from features import engineer_features, get_feature_cols, split_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH    = "models/xgboost_model.pkl"
FEATURES_PATH = "models/feature_cols.pkl"
REPORT_DIR    = "monitoring/reports"
Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)


def load_reference_data() -> pd.DataFrame:
    """
    Load and prepare reference dataset.
    Reference = training data distribution.
    Any drift from this = model may need retraining.
    """
    logger.info("Loading reference data...")
    df      = pd.read_excel("data/data_bnpl.xlsx")
    df_feat = engineer_features(df, is_training=True)
    train, _ = split_data(df_feat)

    features = joblib.load(FEATURES_PATH)
    return train[features].sample(1000, random_state=42)


def load_production_data() -> pd.DataFrame:
    """
    Load recent production predictions for monitoring.
    In real production this comes from your predictions.jsonl log.
    For demo we simulate drift by perturbing the reference data.
    """
    predictions_path = Path("predictions.jsonl")

    if predictions_path.exists():
        # Load real production predictions if available
        records = []
        with open(predictions_path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    records.append(record["input"])
                except:
                    continue

        if len(records) >= 50:
            logger.info(f"Loaded {len(records)} real predictions")
            return pd.DataFrame(records)

    # Simulate production data with slight drift for demo
    logger.info("Simulating production data with drift...")
    reference = load_reference_data()
    production = reference.copy()

    # Simulate drift — customers paying later than training period
    production["days_late_1"]    = production["days_late_1"] * 1.3 + 2
    production["shortfall_1"]    = production["shortfall_1"] * 1.2
    production["log_order_amount"] = production["log_order_amount"] + 0.1

    return production


def run_drift_report(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    report_name: str = None
) -> dict:
    """
    Run Evidently drift report comparing reference vs production.

    Returns:
        Dictionary with drift metrics and whether action needed
    """
    if report_name is None:
        report_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info(f"Running drift report: {report_name}")

    # ── Build Evidently report ────────────────────────────────────────────
    report = Report(metrics=[
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        ColumnDriftMetric(column_name="days_late_1"),
        ColumnDriftMetric(column_name="shortfall_1"),
        ColumnDriftMetric(column_name="log_order_amount"),
        ColumnDriftMetric(column_name="merchant_default_rate"),
        ColumnDriftMetric(column_name="customer_prior_default_rate"),
    ])

    report.run(
        reference_data = reference,
        current_data   = production
    )

    # ── Save HTML report ──────────────────────────────────────────────────
    html_path = f"{REPORT_DIR}/drift_report_{report_name}.html"
    report.save_html(html_path)
    logger.info(f"HTML report saved: {html_path}")

    # ── Extract key metrics ───────────────────────────────────────────────
    report_dict = report.as_dict()
    metrics     = report_dict["metrics"]

    # Dataset-level drift
    dataset_drift   = metrics[0]["result"]
    n_drifted_cols  = dataset_drift.get("number_of_drifted_columns", 0)
    n_total_cols    = dataset_drift.get("number_of_columns", 0)
    dataset_drifted = dataset_drift.get("dataset_drift", False)
    share_drifted   = dataset_drift.get("share_of_drifted_columns", 0)

    # Per-column drift scores
    column_results = {}
    for i, col in enumerate([
        "days_late_1", "shortfall_1", "log_order_amount",
        "merchant_default_rate", "customer_prior_default_rate"
    ]):
        col_metric = metrics[i + 2]["result"]
        column_results[col] = {
            "drifted":   col_metric.get("drift_detected", False),
            "p_value":   col_metric.get("p_value", None),
            "statistic": col_metric.get("statistic", None),
        }

    # ── Determine action required ─────────────────────────────────────────
    action_required = False
    action_reason   = []

    if dataset_drifted and share_drifted > 0.3:
        action_required = True
        action_reason.append(
            f"{n_drifted_cols}/{n_total_cols} features drifted "
            f"({share_drifted*100:.0f}%) — consider retraining"
        )

    if column_results.get("days_late_1", {}).get("drifted"):
        action_required = True
        action_reason.append(
            "days_late_1 drifted — payment behaviour changing"
        )

    if column_results.get("shortfall_1", {}).get("drifted"):
        action_required = True
        action_reason.append(
            "shortfall_1 drifted — payment amounts changing"
        )

    # ── Build summary ─────────────────────────────────────────────────────
    summary = {
        "timestamp":        datetime.utcnow().isoformat(),
        "report_path":      html_path,
        "dataset_drifted":  dataset_drifted,
        "n_drifted_cols":   n_drifted_cols,
        "n_total_cols":     n_total_cols,
        "share_drifted":    round(share_drifted, 4),
        "column_results":   column_results,
        "action_required":  action_required,
        "action_reasons":   action_reason,
        "status":           "ACTION REQUIRED" if action_required else "OK"
    }

    # ── Print summary ─────────────────────────────────────────────────────
    logger.info("=== DRIFT MONITORING SUMMARY ===")
    logger.info(f"  Status:          {summary['status']}")
    logger.info(f"  Dataset drifted: {dataset_drifted}")
    logger.info(f"  Drifted cols:    {n_drifted_cols}/{n_total_cols}")
    logger.info(f"  Share drifted:   {share_drifted*100:.0f}%")
    logger.info("")
    logger.info("  Per-feature drift:")
    for col, result in column_results.items():
        status = "DRIFT" if result["drifted"] else "OK"
        pval   = f"p={result['p_value']:.4f}" if result["p_value"] else ""
        logger.info(f"    {col:<35} {status}  {pval}")

    if action_required:
        logger.info("")
        logger.info("  ACTION REQUIRED:")
        for reason in action_reason:
            logger.info(f"    → {reason}")

    # ── Save summary JSON ─────────────────────────────────────────────────
    summary_path = f"{REPORT_DIR}/summary_{report_name}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def check_psi(reference: pd.DataFrame,
              production: pd.DataFrame,
              column: str,
              n_bins: int = 10) -> float:
    """
    Calculate Population Stability Index for a single feature.
    PSI < 0.10: stable
    PSI 0.10-0.25: moderate shift — investigate
    PSI > 0.25: significant shift — action required
    """
    ref_vals  = reference[column].dropna()
    prod_vals = production[column].dropna()

    # Create bins from reference data
    bins = np.percentile(ref_vals, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)

    if len(bins) < 2:
        return 0.0

    ref_counts  = np.histogram(ref_vals,  bins=bins)[0]
    prod_counts = np.histogram(prod_vals, bins=bins)[0]

    # Avoid division by zero
    ref_pcts  = (ref_counts  + 0.0001) / len(ref_vals)
    prod_pcts = (prod_counts + 0.0001) / len(prod_vals)

    psi = np.sum((prod_pcts - ref_pcts) * np.log(prod_pcts / ref_pcts))
    return float(round(psi, 4))


def run_psi_checks(reference: pd.DataFrame,
                   production: pd.DataFrame) -> dict:
    """
    Run PSI checks on all key features.
    Credit risk standard metric for distribution shift.
    """
    key_features = [
        "days_late_1",
        "shortfall_1",
        "log_order_amount",
        "merchant_default_rate",
        "customer_prior_default_rate",
        "n_missing_payments",
        "amount_bucket",
    ]

    results = {}
    logger.info("=== PSI CHECKS ===")

    for col in key_features:
        if col in reference.columns and col in production.columns:
            psi = check_psi(reference, production, col)

            status = (
                "STABLE"      if psi < 0.10 else
                "INVESTIGATE" if psi < 0.25 else
                "ACTION"
            )
            results[col] = {"psi": psi, "status": status}
            logger.info(f"  {col:<35} PSI={psi:.4f}  {status}")

    return results


if __name__ == "__main__":
    # Load reference and production data
    reference  = load_reference_data()
    production = load_production_data()

    logger.info(f"Reference rows:  {len(reference):,}")
    logger.info(f"Production rows: {len(production):,}")

    # Run PSI checks first (fast)
    psi_results = run_psi_checks(reference, production)

    # Run full Evidently drift report
    summary = run_drift_report(reference, production)

    # Final verdict
    print("\n" + "="*50)
    print(f"MONITORING STATUS: {summary['status']}")
    print(f"HTML Report:       {summary['report_path']}")
    print("="*50)