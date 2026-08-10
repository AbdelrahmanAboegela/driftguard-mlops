"""Integration tests for FastAPI inference endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from serving.app import app
from serving.model_loader import model_manager


@pytest.fixture(scope="module")
def client():
    model_manager.load()
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ["healthy", "degraded"]
    assert data["model_loaded"] is True
    assert "uptime_seconds" in data


def test_predict_single_endpoint(client):
    payload = {
        "Time": 3600.0,
        "Amount": 45.50,
        "request_id": "test_req_001",
    }
    for i in range(1, 29):
        payload[f"V{i}"] = 0.0

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["request_id"] == "test_req_001"
    assert 0.0 <= data["fraud_score"] <= 1.0
    assert isinstance(data["is_fraud"], bool)
    assert data["latency_ms"] >= 0.0


def test_predict_batch_endpoint(client):
    records = []
    for idx in range(5):
        payload = {
            "Time": 100.0 * idx,
            "Amount": 20.0 + idx,
            "request_id": f"batch_{idx}",
        }
        for i in range(1, 29):
            payload[f"V{i}"] = 0.1 * idx
        records.append(payload)

    resp = client.post("/predict/batch", json={"transactions": records})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_processed"] == 5
    assert len(data["predictions"]) == 5


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "driftguard_http_requests_total" in resp.text


def test_reload_model_endpoint_auth(client):
    # Missing/invalid key should return 401
    resp_unauth = client.post("/reload-model", headers={"x-api-key": "invalid-key"})
    assert resp_unauth.status_code == 401

    # Valid key should succeed
    resp_auth = client.post("/reload-model", headers={"x-api-key": "dev-admin-key"})
    assert resp_auth.status_code == 200
    assert resp_auth.json()["status"] == "success"
