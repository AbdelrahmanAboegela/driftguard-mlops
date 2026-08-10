"""FastAPI production serving layer with Prometheus observability for DriftGuard."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.responses import Response

from serving.logger import prediction_logger
from serving.model_loader import model_manager
from serving.schemas import (
    BatchPredictionResponse,
    BatchTransactionSchema,
    DriftStatusResponse,
    HealthResponse,
    PredictionResponse,
    TransactionSchema,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Prometheus Metrics Definitions
HTTP_REQUESTS_TOTAL = Counter(
    "driftguard_http_requests_total",
    "Total HTTP requests handled",
    ["method", "endpoint", "status_code"],
)
INFERENCE_LATENCY_HISTOGRAM = Histogram(
    "driftguard_inference_latency_seconds",
    "Model inference latency distribution in seconds",
    buckets=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0],
)
FRAUD_SCORE_HISTOGRAM = Histogram(
    "driftguard_fraud_score_distribution",
    "Distribution of output fraud risk scores",
    buckets=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)
FRAUD_DECISIONS_TOTAL = Counter(
    "driftguard_fraud_decisions_total",
    "Total fraud vs legitimate classifications",
    ["decision"],
)
ACTIVE_MODEL_VERSION = Gauge(
    "driftguard_active_model_version",
    "Active production model version indicator",
    ["version"],
)
DATASET_DRIFT_SCORE_GAUGE = Gauge(
    "driftguard_dataset_drift_score",
    "Current dataset drift score (Evidently / PSI)",
)
DATASET_DRIFT_ALERT_GAUGE = Gauge(
    "driftguard_dataset_drift_alert",
    "Alert flag: 1 if drift detected, 0 otherwise",
)

START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown initialization."""
    logger.info("Initializing DriftGuard inference service...")
    model_manager.load()
    ACTIVE_MODEL_VERSION._metrics.clear()
    ACTIVE_MODEL_VERSION.labels(version=model_manager.model_version).set(1)
    yield
    logger.info("DriftGuard inference service shutting down.")


app = FastAPI(
    title="DriftGuard Fraud Detection API",
    description="High-performance, drift-aware autonomous fraud detection serving layer.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Returns service health, model version, and uptime."""
    uptime = time.time() - START_TIME
    return HealthResponse(
        status="healthy" if model_manager.model is not None else "degraded",
        model_version=model_manager.model_version,
        model_loaded=model_manager.model is not None,
        uptime_seconds=round(uptime, 2),
        features_count=len(model_manager.transformer.feature_columns) if model_manager.transformer else 0,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_single_transaction(
    transaction: TransactionSchema,
    background_tasks: BackgroundTasks,
) -> PredictionResponse:
    """Computes fraud risk score and binary decision for a single transaction record."""
    t0 = time.perf_counter()
    raw_dict = transaction.model_dump()
    req_id = transaction.request_id or "unknown"

    try:
        score, is_fraud, threshold, model_ver, latency_ms = model_manager.predict_single(raw_dict)
    except Exception as exc:
        HTTP_REQUESTS_TOTAL.labels(method="POST", endpoint="/predict", status_code="500").inc()
        logger.error("Prediction failure for request %s: %s", req_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(exc)}",
        ) from exc

    total_latency_sec = time.perf_counter() - t0

    # Prometheus metrics
    HTTP_REQUESTS_TOTAL.labels(method="POST", endpoint="/predict", status_code="200").inc()
    INFERENCE_LATENCY_HISTOGRAM.observe(total_latency_sec)
    FRAUD_SCORE_HISTOGRAM.observe(score)
    FRAUD_DECISIONS_TOTAL.labels(decision="fraud" if is_fraud else "legitimate").inc()

    # Async feature and prediction logging
    background_tasks.add_task(
        prediction_logger.log_prediction,
        request_id=req_id,
        model_version=model_ver,
        features=raw_dict,
        fraud_score=score,
        is_fraud=is_fraud,
        threshold_used=threshold,
        latency_ms=latency_ms,
    )

    return PredictionResponse(
        request_id=req_id,
        fraud_score=round(score, 6),
        is_fraud=is_fraud,
        threshold_used=round(threshold, 4),
        model_version=model_ver,
        latency_ms=round(latency_ms, 3),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch_transactions(
    payload: BatchTransactionSchema,
    background_tasks: BackgroundTasks,
) -> BatchPredictionResponse:
    """Processes a batch of transactions with high-throughput vectorized scoring."""
    t0 = time.perf_counter()
    raw_list = [t.model_dump() for t in payload.transactions]

    try:
        results, batch_latency_ms = model_manager.predict_batch(raw_list)
    except Exception as exc:
        HTTP_REQUESTS_TOTAL.labels(method="POST", endpoint="/predict/batch", status_code="500").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {str(exc)}",
        ) from exc

    # Metrics and background logging
    log_records = []
    for i, res in enumerate(results):
        score = res["fraud_score"]
        is_fraud = res["is_fraud"]
        FRAUD_SCORE_HISTOGRAM.observe(score)
        FRAUD_DECISIONS_TOTAL.labels(decision="fraud" if is_fraud else "legitimate").inc()

        log_records.append(
            {
                "request_id": res["request_id"],
                "model_version": res["model_version"],
                "features": raw_list[i],
                "fraud_score": score,
                "is_fraud": is_fraud,
                "threshold_used": res["threshold_used"],
                "latency_ms": batch_latency_ms / max(1, len(raw_list)),
            }
        )

    background_tasks.add_task(prediction_logger.log_batch, log_records)
    HTTP_REQUESTS_TOTAL.labels(method="POST", endpoint="/predict/batch", status_code="200").inc()

    formatted_predictions = [
        PredictionResponse(
            request_id=r["request_id"],
            fraud_score=round(r["fraud_score"], 6),
            is_fraud=r["is_fraud"],
            threshold_used=round(r["threshold_used"], 4),
            model_version=r["model_version"],
            latency_ms=round(batch_latency_ms / len(results), 3),
        )
        for r in results
    ]

    return BatchPredictionResponse(
        predictions=formatted_predictions,
        total_processed=len(formatted_predictions),
        batch_latency_ms=round(batch_latency_ms, 3),
    )


@app.post("/reload-model", tags=["Model Management"])
async def reload_model() -> dict[str, str]:
    """Hot-reloads the production model from MLflow Registry or local store."""
    try:
        model_manager.load(force_reload=True)
        ACTIVE_MODEL_VERSION._metrics.clear()
        ACTIVE_MODEL_VERSION.labels(version=model_manager.model_version).set(1)
        logger.info("Hot-reloaded model version: %s", model_manager.model_version)
        return {
            "status": "success",
            "message": f"Successfully reloaded model {model_manager.model_version}",
            "model_version": model_manager.model_version,
        }
    except Exception as exc:
        logger.error("Hot-reload failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload model: {str(exc)}",
        ) from exc


@app.get("/drift/status", response_model=DriftStatusResponse, tags=["Monitoring"])
async def get_drift_status() -> DriftStatusResponse:
    """Returns the latest evaluated drift metrics."""
    from monitoring.drift_detector import get_latest_drift_summary

    summary = get_latest_drift_summary()
    return DriftStatusResponse(
        drift_detected=summary["drift_detected"],
        dataset_drift_score=summary["dataset_drift_score"],
        max_feature_psi=summary["max_feature_psi"],
        drifted_features=summary["drifted_features"],
        sample_window_size=summary["sample_window_size"],
        last_evaluated_timestamp=summary["last_evaluated_timestamp"],
    )


@app.get("/metrics", tags=["Observability"])
async def get_metrics() -> Response:
    """Standard Prometheus scraping endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
