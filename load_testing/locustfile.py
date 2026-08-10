"""Locust load testing scenario for DriftGuard inference service."""

from __future__ import annotations

import random
import uuid

from locust import HttpUser, between, task


class FraudDetectionUser(HttpUser):
    """Simulates production traffic from payment gateways."""

    wait_time = between(0.01, 0.05)

    def _generate_transaction(self) -> dict:
        """Generates a realistic transaction payload."""
        is_fraud_sim = random.random() < 0.02

        time_val = random.uniform(0, 172800)
        if is_fraud_sim:
            amount = round(random.uniform(50.0, 1200.0), 2)
            v4 = random.gauss(3.5, 1.0)
            v11 = random.gauss(3.0, 0.8)
            v14 = random.gauss(-5.0, 1.2)
        else:
            amount = round(random.lognormvariate(3.2, 1.2), 2)
            v4 = random.gauss(0.0, 1.0)
            v11 = random.gauss(0.0, 1.0)
            v14 = random.gauss(0.0, 1.0)

        payload = {
            "Time": time_val,
            "Amount": max(0.5, min(amount, 10000.0)),
            "request_id": str(uuid.uuid4()),
        }

        for i in range(1, 29):
            if i == 4:
                payload["V4"] = v4
            elif i == 11:
                payload["V11"] = v11
            elif i == 14:
                payload["V14"] = v14
            else:
                payload[f"V{i}"] = random.gauss(0.0, 1.0)

        return payload

    @task(80)
    def predict_single(self) -> None:
        """Single transaction inference call."""
        payload = self._generate_transaction()
        with self.client.post("/predict", json=payload, catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Status code: {resp.status_code}")

    @task(15)
    def predict_batch(self) -> None:
        """Batch transaction inference call."""
        batch_size = random.randint(5, 25)
        payload = {"transactions": [self._generate_transaction() for _ in range(batch_size)]}
        with self.client.post("/predict/batch", json=payload, catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Batch status code: {resp.status_code}")

    @task(5)
    def health_check(self) -> None:
        """Health status check."""
        self.client.get("/health")
