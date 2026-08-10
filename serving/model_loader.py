"""Model loader and hot-reloading manager for DriftGuard."""

from __future__ import annotations

import json
import logging
import os
import threading
import time

import joblib
import mlflow
import mlflow.xgboost
import pandas as pd

from training.feature_engineering import ARTIFACTS_DIR, FeatureTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_REGISTRY_NAME = os.getenv("MLFLOW_MODEL_NAME", "driftguard-fraud")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")


class ModelManager:
    """Manages thread-safe model loading, inference execution, and dynamic hot-reloading."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.model = None
        self.transformer: FeatureTransformer | None = None
        self.model_version: str = "unknown"
        self.threshold: float = 0.5
        self.metrics: dict = {}
        self.loaded_at: float = 0.0

    def load(self, force_reload: bool = False) -> bool:
        """Loads or reloads the active production model and feature transformer."""
        with self.lock:
            if self.model is not None and not force_reload:
                return True

            logger.info("Initializing model and feature transformer...")

            # 1. Load Feature Transformer
            preprocessor_path = ARTIFACTS_DIR / "preprocessor.joblib"
            if preprocessor_path.exists():
                self.transformer = FeatureTransformer.load(preprocessor_path)
                logger.info("Loaded FeatureTransformer from %s", preprocessor_path)
            else:
                logger.warning(
                    "FeatureTransformer artifact not found at %s. Creating and fitting fallback...",
                    preprocessor_path,
                )
                from training.train import train_baseline_model

                train_baseline_model()
                self.transformer = FeatureTransformer.load(preprocessor_path)

            # 2. Try loading from MLflow Registry only when one is explicitly configured.
            # Local serving intentionally uses the bundled champion artifact; attempting to
            # initialize an implicit SQLite store makes a non-root container block on retries.
            loaded_from_mlflow = False
            if MLFLOW_TRACKING_URI:
                try:
                    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
                    client = mlflow.tracking.MlflowClient()

                    # Try alias 'production' or stage 'Production'
                    model_uri = None
                    try:
                        alias_model = client.get_model_version_by_alias(
                            MODEL_REGISTRY_NAME, "production"
                        )
                        model_uri = f"models:/{MODEL_REGISTRY_NAME}@production"
                        self.model_version = f"v{alias_model.version}"
                    except Exception:
                        # Fallback to stage using search_model_versions
                        filter_string = f"name='{MODEL_REGISTRY_NAME}'"
                        versions = client.search_model_versions(filter_string)
                        production_versions = [
                            v for v in versions if v.current_stage == "Production"
                        ]
                        if production_versions:
                            # Sort by version number descending
                            latest_prod = sorted(
                                production_versions, key=lambda v: int(v.version), reverse=True
                            )[0]
                            model_uri = f"models:/{MODEL_REGISTRY_NAME}/Production"
                            self.model_version = f"v{latest_prod.version}"

                    if model_uri:
                        logger.info(
                            "Loading production model from MLflow Registry: %s...", model_uri
                        )
                        self.model = mlflow.xgboost.load_model(model_uri)
                        loaded_from_mlflow = True
                        logger.info(
                            "Successfully loaded MLflow model version: %s", self.model_version
                        )
                except Exception as exc:
                    logger.warning(
                        "Could not load from MLflow Model Registry (%s). Falling back to local artifact.",
                        exc,
                    )

            # 3. Fallback to local artifact
            if not loaded_from_mlflow:
                local_path = ARTIFACTS_DIR / "champion_model.joblib"
                metadata_path = ARTIFACTS_DIR / "model_metadata.json"

                if not local_path.exists():
                    logger.info("No local artifact found. Triggering baseline model training...")
                    from training.train import train_baseline_model

                    train_baseline_model()

                payload = joblib.load(local_path)
                self.model = payload["model"]
                self.threshold = payload.get("threshold", 0.5)
                self.metrics = payload.get("metrics", {})

                if metadata_path.exists():
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        self.model_version = f"run_{meta.get('run_id', 'local')[:8]}"
                        self.threshold = meta.get("threshold", self.threshold)
                else:
                    self.model_version = f"run_{payload.get('run_id', 'local')[:8]}"

                logger.info(
                    "Loaded model from local artifact: %s (Threshold=%.4f)",
                    self.model_version,
                    self.threshold,
                )

            self.loaded_at = time.time()
            return True

    def predict_single(self, transaction: dict) -> tuple[float, bool, float, str, float]:
        """Runs single-transaction inference.

        Returns:
            (fraud_score, is_fraud, threshold_used, model_version, latency_ms)
        """
        start_time = time.perf_counter()

        if self.model is None or self.transformer is None:
            self.load()

        with self.lock:
            # Transform features
            X_df = self.transformer.transform_single(transaction)

            # Score
            prob = float(self.model.predict_proba(X_df)[0, 1])
            is_fraud = bool(prob >= self.threshold)
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            return prob, is_fraud, self.threshold, self.model_version, latency_ms

    def predict_batch(self, transactions: list[dict]) -> tuple[list[dict], float]:
        """Runs batch inference efficiently."""
        start_time = time.perf_counter()

        if self.model is None or self.transformer is None:
            self.load()

        df_raw = pd.DataFrame(transactions)

        with self.lock:
            X_df = self.transformer.transform(df_raw)
            probs = self.model.predict_proba(X_df)[:, 1]

            results = []
            for i, prob in enumerate(probs):
                p_val = float(prob)
                results.append(
                    {
                        "request_id": transactions[i].get("request_id", f"batch_{i}"),
                        "fraud_score": p_val,
                        "is_fraud": bool(p_val >= self.threshold),
                        "threshold_used": self.threshold,
                        "model_version": self.model_version,
                        "latency_ms": 0.0,
                    }
                )

            total_latency_ms = (time.perf_counter() - start_time) * 1000.0
            return results, total_latency_ms


# Global model manager instance
model_manager = ModelManager()
