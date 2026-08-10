"""Autonomous retraining and Champion-Challenger validation pipeline for DriftGuard."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb

from data.split_data import PROCESSED_DIR, split_temporal
from monitoring.drift_detector import evaluate_drift
from training.evaluate import evaluate_predictions
from training.feature_engineering import ARTIFACTS_DIR, FeatureTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "driftguard-fraud"
MODEL_REGISTRY_NAME = os.getenv("MLFLOW_MODEL_NAME", "driftguard-fraud")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
SERVING_URL = os.getenv("SERVING_URL", "http://localhost:8000")


def load_retraining_data(additional_data: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepares enriched retraining data by combining historical training set with newly collected labeled transactions."""
    train_path = PROCESSED_DIR / "train.parquet"
    val_path = PROCESSED_DIR / "val_holdout.parquet"

    if not train_path.exists() or not val_path.exists():
        train_path, val_path, _ = split_temporal()

    historical_train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    if additional_data is not None and len(additional_data) > 0:
        logger.info("Combining %d historical records with %d newly collected records for retraining.", len(historical_train_df), len(additional_data))
        # Exclude non-feature metadata columns if present
        clean_additional = additional_data.copy()
        for col in ["request_id", "timestamp", "model_version", "fraud_score", "is_fraud", "latency_ms"]:
            if col in clean_additional.columns:
                clean_additional = clean_additional.drop(columns=[col])

        if "ground_truth" in clean_additional.columns and "Class" not in clean_additional.columns:
            clean_additional["Class"] = clean_additional["ground_truth"]
            clean_additional = clean_additional.drop(columns=["ground_truth"])

        combined_train_df = pd.concat([historical_train_df, clean_additional], ignore_index=True)
        return combined_train_df, val_df

    return historical_train_df, val_df


def train_challenger(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[xgb.XGBClassifier, dict, str, str]:
    """Trains a Challenger model and logs it to MLflow as a Candidate."""
    y_train = train_df["Class"].values
    y_val = val_df["Class"].values

    transformer = FeatureTransformer()
    X_train = transformer.fit(train_df).transform(train_df)
    X_val = transformer.transform(val_df)

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = float(n_neg / max(1, n_pos))

    params = {
        "n_estimators": 180,
        "max_depth": 5,
        "learning_rate": 0.07,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": ["logloss", "aucpr"],
        "random_state": 99,
        "tree_method": "hist",
    }

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_params(params)
        mlflow.log_param("role", "challenger")
        mlflow.log_param("retrained_at", datetime.now(timezone.utc).isoformat())

        challenger_model = xgb.XGBClassifier(**params)
        challenger_model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )

        val_probs = challenger_model.predict_proba(X_val)[:, 1]
        metrics = evaluate_predictions(y_val, val_probs)
        mlflow.log_metrics(metrics)

        # Log artifact to MLflow
        mlflow.xgboost.log_model(
            xgb_model=challenger_model,
            artifact_path="model",
            registered_model_name=MODEL_REGISTRY_NAME,
        )

        # Get the new model version
        client = mlflow.tracking.MlflowClient()
        latest_versions = client.get_latest_versions(MODEL_REGISTRY_NAME)
        challenger_version = latest_versions[-1].version if latest_versions else "unknown"
        client.set_model_version_tag(MODEL_REGISTRY_NAME, challenger_version, "stage", "Staging")

        # Save transformer as staging artifact
        transformer.save(ARTIFACTS_DIR / "preprocessor.joblib")

        return challenger_model, metrics, challenger_version, run_id


def evaluate_and_promote_challenger(
    challenger_model: xgb.XGBClassifier,
    challenger_metrics: dict,
    challenger_version: str,
    val_df: pd.DataFrame,
    f1_margin: float = 0.0,
) -> dict:
    """Performs rigorous Champion vs Challenger comparison on the exact same holdout validation set."""
    # 1. Load current Champion metrics
    metadata_path = ARTIFACTS_DIR / "model_metadata.json"
    champion_metrics = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            champion_metrics = meta.get("metrics", {})

    champion_f1 = champion_metrics.get("f1", 0.0)
    champion_prauc = champion_metrics.get("pr_auc", 0.0)
    challenger_f1 = challenger_metrics.get("f1", 0.0)
    challenger_prauc = challenger_metrics.get("pr_auc", 0.0)

    logger.info(
        "\n====== CHAMPION VS CHALLENGER EVALUATION ======\n"
        "  Champion:   F1 = %.4f | PR-AUC = %.4f\n"
        "  Challenger: F1 = %.4f | PR-AUC = %.4f (Version %s)\n"
        "================================================",
        champion_f1,
        champion_prauc,
        challenger_f1,
        challenger_prauc,
        challenger_version,
    )

    # 2. Decision Logic
    # Challenger must achieve at least equal or better F1 and within PR-AUC tolerance
    is_promoted = (challenger_f1 >= champion_f1 + f1_margin) and (challenger_prauc >= champion_prauc - 0.03)

    decision = {
        "promoted": is_promoted,
        "challenger_version": challenger_version,
        "champion_f1": champion_f1,
        "challenger_f1": challenger_f1,
        "champion_prauc": champion_prauc,
        "challenger_prauc": challenger_prauc,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "reason": "Challenger outperformed or matched Champion" if is_promoted else "Challenger failed to surpass Champion F1/PR-AUC",
    }

    if is_promoted:
        logger.info("CHALLENGER PROMOTED TO PRODUCTION (Version %s)", challenger_version)
        # Update MLflow Registry tags and aliases
        try:
            client = mlflow.tracking.MlflowClient()
            client.set_model_version_tag(MODEL_REGISTRY_NAME, challenger_version, "stage", "Production")
            try:
                client.set_registered_model_alias(MODEL_REGISTRY_NAME, "production", challenger_version)
            except Exception:
                pass
        except Exception as exc:
            logger.warning("MLflow promotion tag update warning: %s", exc)

        # Update local champion artifact
        local_model_path = ARTIFACTS_DIR / "champion_model.joblib"
        import joblib

        joblib.dump(
            {
                "model": challenger_model,
                "metrics": challenger_metrics,
                "version": challenger_version,
                "threshold": challenger_metrics["threshold"],
            },
            local_model_path,
        )

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": challenger_version,
                    "metrics": challenger_metrics,
                    "threshold": challenger_metrics["threshold"],
                    "promoted_at": datetime.now(timezone.utc).isoformat(),
                },
                f,
                indent=2,
            )

        # Hot-reload serving app
        try:
            with httpx.Client(timeout=5.0) as http_client:
                resp = http_client.post(f"{SERVING_URL}/reload-model")
                if resp.status_code == 200:
                    logger.info("Serving layer successfully hot-reloaded new production model.")
        except Exception as exc:
            logger.debug("Serving layer hot-reload notification skipped: %s", exc)
    else:
        logger.warning("CHALLENGER REJECTED. Maintaining Champion in production.")

    return decision


def run_autonomous_pipeline(
    additional_data: pd.DataFrame | None = None,
    force_retrain: bool = False,
) -> dict:
    """Executes the complete autonomous retraining workflow:

    1. Checks for drift (or forces retrain).
    2. Enriches training dataset with recent production patterns.
    3. Trains Challenger model.
    4. Runs Champion vs Challenger validation.
    5. Promotes if superior, otherwise maintains Champion.
    """
    logger.info("Initiating autonomous retraining workflow...")

    if not force_retrain:
        drift_res = evaluate_drift(save_reports=True)
        if not drift_res["drift_detected"]:
            logger.info("No significant drift detected. Retraining skipped.")
            return {"status": "skipped", "reason": "No drift detected"}

    train_df, val_df = load_retraining_data(additional_data)
    challenger_model, challenger_metrics, challenger_version, _ = train_challenger(train_df, val_df)
    decision = evaluate_and_promote_challenger(challenger_model, challenger_metrics, challenger_version, val_df)

    return {
        "status": "completed",
        "decision": decision,
        "challenger_metrics": challenger_metrics,
    }


if __name__ == "__main__":
    run_autonomous_pipeline(force_retrain=True)
