"""Feature and prediction audit logging module for DriftGuard."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_DIR = Path(__file__).resolve().parent / "storage"
DB_PATH = DB_DIR / "predictions.db"


class PredictionLogger:
    """Manages audit logging of features, model decisions, and ground-truth labels."""

    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        """Initializes SQLite schema for feature/prediction logs."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    fraud_score REAL NOT NULL,
                    is_fraud INTEGER NOT NULL,
                    threshold_used REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    ground_truth INTEGER DEFAULT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON prediction_logs(timestamp);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_request_id ON prediction_logs(request_id);")

    def log_prediction(
        self,
        request_id: str,
        model_version: str,
        features: dict,
        fraud_score: float,
        is_fraud: bool,
        threshold_used: float,
        latency_ms: float,
        ground_truth: int | None = None,
    ) -> None:
        """Persists a prediction event into the database."""
        now_utc = datetime.now(timezone.utc).isoformat()
        features_str = json.dumps(features)

        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO prediction_logs (
                        request_id, timestamp, model_version, features_json,
                        fraud_score, is_fraud, threshold_used, latency_ms, ground_truth
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        request_id,
                        now_utc,
                        model_version,
                        features_str,
                        fraud_score,
                        int(is_fraud),
                        threshold_used,
                        latency_ms,
                        ground_truth,
                    ),
                )
        except Exception as exc:
            logger.error("Failed to log prediction %s: %s", request_id, exc)

    def log_batch(self, records: list[dict]) -> None:
        """Persists a batch of prediction events."""
        now_utc = datetime.now(timezone.utc).isoformat()
        data_tuples = [
            (
                r["request_id"],
                now_utc,
                r["model_version"],
                json.dumps(r["features"]),
                r["fraud_score"],
                int(r["is_fraud"]),
                r["threshold_used"],
                r["latency_ms"],
                r.get("ground_truth"),
            )
            for r in records
        ]

        try:
            with self._get_connection() as conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO prediction_logs (
                        request_id, timestamp, model_version, features_json,
                        fraud_score, is_fraud, threshold_used, latency_ms, ground_truth
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    data_tuples,
                )
        except Exception as exc:
            logger.error("Failed to log batch predictions: %s", exc)

    def get_recent_dataframe(self, limit: int = 2000) -> pd.DataFrame:
        """Retrieves recent logged transactions as a structured pandas DataFrame."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT request_id, timestamp, model_version, features_json,
                       fraud_score, is_fraud, latency_ms, ground_truth
                FROM prediction_logs
                ORDER BY id DESC
                LIMIT ?;
                """,
                (limit,),
            )
            rows = cursor.fetchall()

        if not rows:
            return pd.DataFrame()

        records = []
        for row in reversed(rows):
            req_id, ts, mv, feats_json, score, is_f, lat, gt = row
            feats = json.loads(feats_json)
            feats["request_id"] = req_id
            feats["timestamp"] = ts
            feats["model_version"] = mv
            feats["fraud_score"] = score
            feats["is_fraud"] = is_f
            feats["latency_ms"] = lat
            feats["ground_truth"] = gt
            records.append(feats)

        return pd.DataFrame(records)

    def count_total_logs(self) -> int:
        """Returns the total number of logged predictions."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM prediction_logs;")
            return cursor.fetchone()[0]


# Global logger instance
prediction_logger = PredictionLogger()
