"""Unit tests for statistical drift detection and PSI calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monitoring.drift_detector import calculate_psi, compute_feature_drift_stats


def test_calculate_psi_identical_distributions():
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=0.0, scale=1.0, size=2000)
    curr = rng.normal(loc=0.0, scale=1.0, size=2000)

    psi = calculate_psi(ref, curr)
    assert psi < 0.10, f"Expected PSI < 0.10 for identical distributions, got {psi}"


def test_calculate_psi_shifted_distribution():
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=0.0, scale=1.0, size=2000)
    # Substantial distribution shift
    curr = rng.normal(loc=2.5, scale=1.5, size=2000)

    psi = calculate_psi(ref, curr)
    assert psi >= 0.25, f"Expected critical PSI >= 0.25 for shifted distribution, got {psi}"


def test_compute_feature_drift_stats():
    rng = np.random.default_rng(123)
    n = 500

    ref_data = {"Amount": rng.exponential(scale=50.0, size=n), "V1": rng.normal(0, 1, size=n), "V2": rng.normal(0, 1, size=n)}
    curr_data = {
        "Amount": rng.exponential(scale=250.0, size=n),  # Significant drift
        "V1": rng.normal(3.0, 1, size=n),  # Significant drift
        "V2": rng.normal(0, 1, size=n),  # Stable
    }

    ref_df = pd.DataFrame(ref_data)
    curr_df = pd.DataFrame(curr_data)

    stats = compute_feature_drift_stats(ref_df, curr_df)
    assert "Amount" in stats["drifted_features"]
    assert "V1" in stats["drifted_features"]
    assert stats["max_feature_psi"] >= 0.25
    assert stats["share_drifted"] >= 0.5
