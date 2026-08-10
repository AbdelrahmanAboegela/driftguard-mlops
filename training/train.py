"""Model training pipeline with MLflow tracking & registry integration for DriftGuard."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb

from data.split_data import PROCESSED_DIR, split_temporal
from training.evaluate import evaluate_predictions
from training.feature_engineering import ARTIFACTS_DIR, FeatureTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "driftguard-fraud"
MODEL_REGISTRY_NAME = "driftguard-fraud"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def train_baseline_model(
    train_path: Path | None = None,
    val_path: Path | None = None,
    tag_as_production: bool = True,
    model_params: dict | None = None,
) -> tuple[xgb.XGBClassifier, dict, str]:
    """Trains an XGBoost fraud detection model, tracks experiment in MLflow, and registers the model.

    Args:
        train_path: Path to training parquet dataset.
        val_path: Path to validation holdout parquet dataset.
        tag_as_production: If True, tags the registered model as 'Production' (Champion).
        model_params: Optional hyperparameters overriding defaults.

    Returns:
        (trained_model, metrics_dict, mlflow_run_id)
    """
    # 1. Ensure data exists
    if train_path is None or not train_path.exists():
        train_path = PROCESSED_DIR / "train.parquet"
        val_path = PROCESSED_DIR / "val_holdout.parquet"
        if not train_path.exists() or not val_path.exists():
            train_path, val_path, _ = split_temporal()

    logger.info("Loading training data from %s and validation data from %s...", train_path, val_path)
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    y_train = train_df["Class"].values
    y_val = val_df["Class"].values

    # 2. Feature Engineering & Scaling
    transformer = FeatureTransformer()
    X_train = transformer.fit(train_df).transform(train_df)
    X_val = transformer.transform(val_df)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    transformer.save(ARTIFACTS_DIR / "preprocessor.joblib")

    # 3. Calculate class imbalance weight
    # scale_pos_weight = total_negative / total_positive
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    computed_scale_pos_weight = float(n_neg / max(1, n_pos))

    logger.info(
        "Training distribution: %d negative, %d positive -> scale_pos_weight=%.2f",
        n_neg,
        n_pos,
        computed_scale_pos_weight,
    )

    # 4. Model hyperparameters
    default_params = {
        "n_estimators": 160,
        "max_depth": 5,
        "learning_rate": 0.08,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "scale_pos_weight": computed_scale_pos_weight,
        "eval_metric": ["logloss", "aucpr"],
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    if model_params:
        default_params.update(model_params)

    # 5. MLflow Tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("Started MLflow Run ID: %s (Experiment: %s)", run_id, EXPERIMENT_NAME)

        # Log params
        mlflow.log_params(default_params)
        mlflow.log_param("train_samples", len(train_df))
        mlflow.log_param("val_samples", len(val_df))
        mlflow.log_param("class_imbalance_ratio", f"1:{int(computed_scale_pos_weight)}")

        # 6. Train Model
        model = xgb.XGBClassifier(**default_params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )

        # 7. Evaluate on Frozen Holdout Validation Set
        val_probs = model.predict_proba(X_val)[:, 1]
        metrics = evaluate_predictions(y_val, val_probs)

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)

        # 8. Log Model and Register in MLflow Registry
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name=MODEL_REGISTRY_NAME,
        )

        # Save local fallback artifact for robust standalone serving
        local_model_path = ARTIFACTS_DIR / "champion_model.joblib"
        joblib.dump(
            {
                "model": model,
                "metrics": metrics,
                "run_id": run_id,
                "threshold": metrics["threshold"],
            },
            local_model_path,
        )
        metadata_path = ARTIFACTS_DIR / "model_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": run_id,
                    "metrics": metrics,
                    "threshold": metrics["threshold"],
                    "is_production": tag_as_production,
                },
                f,
                indent=2,
            )

        logger.info("Saved local fallback model to %s and metadata to %s", local_model_path, metadata_path)

        # 9. Model Registry Tagging / Aliasing
        try:
            client = mlflow.tracking.MlflowClient()
            # Get latest version registered
            latest_versions = client.get_latest_versions(MODEL_REGISTRY_NAME)
            if latest_versions:
                latest_v = latest_versions[-1].version
                tag_name = "Production" if tag_as_production else "Staging"
                client.set_model_version_tag(MODEL_REGISTRY_NAME, latest_v, "stage", tag_name)
                # Set alias for modern MLflow client
                try:
                    alias_name = "production" if tag_as_production else "staging"
                    client.set_registered_model_alias(MODEL_REGISTRY_NAME, alias_name, latest_v)
                except Exception as alias_err:
                    logger.debug("Alias setting skipped: %s", alias_err)
                logger.info("Successfully tagged Model Version %s as '%s'", latest_v, tag_name)
        except Exception as reg_err:
            logger.warning("MLflow registry tagging warning: %s", reg_err)

    return model, metrics, run_id


if __name__ == "__main__":
    train_baseline_model()
