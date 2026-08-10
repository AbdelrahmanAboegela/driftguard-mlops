"""Integration tests for autonomous retraining pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from orchestration.retrain_pipeline import evaluate_and_promote_challenger, train_challenger


def _create_mock_data(n=200):
    rng = np.random.default_rng(42)
    data = {
        "Time": rng.uniform(0, 172800, size=n),
        "Amount": rng.exponential(scale=50.0, size=n),
        "Class": rng.choice([0, 1], size=n, p=[0.92, 0.08]),
    }
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0, 1, size=n)
    return pd.DataFrame(data)


def test_train_challenger_and_evaluate(tmp_path, monkeypatch):
    monkeypatch.setattr("orchestration.retrain_pipeline.ARTIFACTS_DIR", tmp_path)
    train_df = _create_mock_data(250)
    val_df = _create_mock_data(80)

    model, metrics, version, run_id = train_challenger(train_df, val_df)

    assert model is not None
    assert "f1" in metrics
    assert "pr_auc" in metrics
    assert version is not None

    decision = evaluate_and_promote_challenger(
        challenger_model=model,
        challenger_metrics=metrics,
        challenger_version=version,
        val_df=val_df,
    )

    assert "promoted" in decision
    assert "reason" in decision
