import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import optuna
import logging
import os
from pathlib import Path

from features import engineer_features, get_feature_cols, split_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH  = "data/data_bnpl.xlsx"
MODEL_DIR  = "models"
Path(MODEL_DIR).mkdir(exist_ok=True)


def gini(y_true, y_pred_proba) -> float:
    """Gini = 2*AUC - 1. Standard credit risk metric."""
    return 2 * roc_auc_score(y_true, y_pred_proba) - 1


def ks_stat(y_true, y_pred_proba) -> float:
    """KS statistic — max separation between good and bad score distributions."""
    from scipy.stats import ks_2samp
    scores_good = y_pred_proba[y_true == 0]
    scores_bad  = y_pred_proba[y_true == 1]
    ks, _       = ks_2samp(scores_good, scores_bad)
    return ks


def train(use_smote: bool = True,
          n_optuna_trials: int = 30) -> str:
    """
    Full training pipeline:
    1. Load and engineer features
    2. Time-based train/test split
    3. SMOTE for class imbalance
    4. Optuna hyperparameter tuning
    5. Final model training
    6. MLflow logging
    7. Model saved to disk
    """
    # ── Load data ────────────────────────────────────────────────────────
    logger.info("Loading data...")
    df = pd.read_excel(DATA_PATH)

    # ── Feature engineering ───────────────────────────────────────────────
    df_feat = engineer_features(df, is_training=True)
    FEATURES = get_feature_cols()

    # ── Train/test split (time-based) ────────────────────────────────────
    train_df, test_df = split_data(df_feat)

    X_train = train_df[FEATURES]
    y_train = train_df["target"]
    X_test  = test_df[FEATURES]
    y_test  = test_df["target"]

    logger.info(f"Train shape: {X_train.shape}")
    logger.info(f"Test shape:  {X_test.shape}")

    # ── SMOTE for class imbalance ─────────────────────────────────────────
    if use_smote:
        logger.info("Applying SMOTE...")
        smote = SMOTE(
            sampling_strategy=0.2,  # Upsample minority to 20% of majority
            random_state=42
        )
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logger.info(f"After SMOTE — Train shape: {X_train.shape}")
        logger.info(f"After SMOTE — Default rate: {y_train.mean()*100:.1f}%")

    # ── Optuna hyperparameter tuning ──────────────────────────────────────
    logger.info(f"Running Optuna tuning ({n_optuna_trials} trials)...")

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0, 5),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0, 2),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0, 2),
            "scale_pos_weight":  trial.suggest_float("scale_pos_weight", 1, 25),
            "eval_metric":       "auc",
            "random_state":      42,
            "use_label_encoder": False,
        }
        # NEW
        model = XGBClassifier(**params, early_stopping_rounds=20)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
            )
        proba = model.predict_proba(X_test)[:, 1]
        return gini(y_test, proba)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_optuna_trials)

    best_params = study.best_params
    logger.info(f"Best Gini from tuning: {study.best_value:.4f}")
    logger.info(f"Best params: {best_params}")

    # ── Final model training ──────────────────────────────────────────────
    mlflow.set_experiment("bnpl-default-prediction")

    with mlflow.start_run(run_name="xgboost-final") as run:

        best_params.update({
            "eval_metric":       "auc",
            "random_state":      42,
            "use_label_encoder": False,
        })

        model = XGBClassifier(**best_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # ── Evaluate ─────────────────────────────────────────────────────
        proba  = model.predict_proba(X_test)[:, 1]
        preds  = (proba >= 0.3).astype(int)   # Lower threshold for recall

        metrics = {
            "gini":      gini(y_test, proba),
            "ks_stat":   ks_stat(y_test, proba),
            "auc_roc":   roc_auc_score(y_test, proba),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall":    recall_score(y_test, preds, zero_division=0),
            "f1":        f1_score(y_test, preds, zero_division=0),
        }

        logger.info("=== MODEL PERFORMANCE ===")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        # ── MLflow logging ────────────────────────────────────────────────
        mlflow.log_params(best_params)
        mlflow.log_params({"use_smote": use_smote,
                           "n_features": len(FEATURES),
                           "train_size": len(X_train),
                           "test_size":  len(X_test)})
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "model")

        # ── Save model and feature list locally ───────────────────────────
        model_path    = f"{MODEL_DIR}/xgboost_model.pkl"
        features_path = f"{MODEL_DIR}/feature_cols.pkl"

        joblib.dump(model,    model_path)
        joblib.dump(FEATURES, features_path)

        logger.info(f"Model saved to: {model_path}")
        logger.info(f"MLflow run ID:  {run.info.run_id}")

        return model_path


if __name__ == "__main__":
    model_path = train(use_smote=True, n_optuna_trials=30)
    logger.info(f"Training complete. Model at: {model_path}")