"""Tests for imbalance controls and post-selection test reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from training.evaluate import bootstrap_confidence_intervals, evaluate_predictions
from training.imbalance import resample_training_data


def test_smote_balances_only_the_supplied_training_fold():
    features = pd.DataFrame({"feature": np.arange(30, dtype=float)})
    labels = np.array([0] * 24 + [1] * 6)

    resampled_features, resampled_labels = resample_training_data(features, labels, method="smote")

    assert len(resampled_features) == len(resampled_labels)
    assert int((resampled_labels == 0).sum()) == int((resampled_labels == 1).sum())
    assert len(labels) == 30


def test_cost_metrics_and_confidence_intervals_are_reported():
    labels = np.array([0] * 80 + [1] * 20)
    probabilities = np.concatenate([np.linspace(0.01, 0.45, 80), np.linspace(0.4, 0.99, 20)])

    metrics = evaluate_predictions(
        labels,
        probabilities,
        threshold=0.5,
        false_positive_cost=1.0,
        false_negative_cost=25.0,
    )
    intervals = bootstrap_confidence_intervals(labels, probabilities, threshold=0.5, iterations=20)

    assert 0.0 <= metrics["g_mean"] <= 1.0
    assert metrics["expected_cost"] >= 0.0
    assert intervals["pr_auc"]["lower"] <= intervals["pr_auc"]["upper"]
