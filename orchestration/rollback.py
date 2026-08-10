"""Disaster recovery and model rollback manager for DriftGuard."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
import httpx
import mlflow

from training.feature_engineering import ARTIFACTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_REGISTRY_NAME = os.getenv("MLFLOW_MODEL_NAME", "driftguard-fraud")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
SERVING_URL = os.getenv("SERVING_URL", "http://localhost:8000")


def rollback_production_model(target_version: str | int | None = None) -> bool:
    """Rolls back the production model in MLflow Registry and local artifacts to a previous stable version.

    If target_version is not specified, selects the immediate predecessor of the current Production version.
    """
    logger.info("Initiating model rollback procedure...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    try:
        all_versions = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")
        if not all_versions:
            logger.error("No registered model versions found for %s", MODEL_REGISTRY_NAME)
            return False

        # Sort versions numerically
        sorted_versions = sorted(all_versions, key=lambda v: int(v.version))

        if target_version is None:
            # Find the previous version before the latest
            if len(sorted_versions) < 2:
                logger.warning("Only 1 version exists (Version %s). Rollback not possible.", sorted_versions[0].version)
                return False
            target_v_obj = sorted_versions[-2]
            target_v = target_v_obj.version
        else:
            target_v = str(target_version)
            matching = [v for v in sorted_versions if v.version == target_v]
            if not matching:
                logger.error("Specified target version %s not found in MLflow registry.", target_v)
                return False

        logger.info("Rolling back Production stage to Model Version %s...", target_v)

        # 1. Update MLflow Registry tag and alias
        client.set_model_version_tag(MODEL_REGISTRY_NAME, target_v, "stage", "Production")
        try:
            client.set_registered_model_alias(MODEL_REGISTRY_NAME, "production", target_v)
        except Exception:
            pass

        # 2. Update local metadata
        metadata_path = ARTIFACTS_DIR / "model_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": target_v,
                    "rolled_back_at": datetime.now(timezone.utc).isoformat(),
                    "status": "rolled_back",
                },
                f,
                indent=2,
            )

        # 3. Notify serving layer
        try:
            with httpx.Client(timeout=5.0) as http_client:
                resp = http_client.post(f"{SERVING_URL}/reload-model")
                if resp.status_code == 200:
                    logger.info("Serving layer reloaded rolled back model version %s.", target_v)
        except Exception as exc:
            logger.debug("Serving layer notification: %s", exc)

        logger.info("✅ SUCCESS: Rolled back production model to Version %s", target_v)
        return True

    except Exception as exc:
        logger.error("Rollback procedure failed: %s", exc)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="DriftGuard Model Rollback Utility")
    parser.add_argument("--version", type=str, default=None, help="Target model version number to restore")
    args = parser.parse_args()
    rollback_production_model(args.version)


if __name__ == "__main__":
    main()
