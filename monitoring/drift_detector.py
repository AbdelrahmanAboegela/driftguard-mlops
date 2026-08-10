"""Evidently AI and statistical drift detection engine for DriftGuard."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from data.split_data import PROCESSED_DIR, split_temporal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
PSI_THRESHOLD_CRITICAL = 0.25  # Industry standard: >0.25 indicates significant distribution shift
PSI_THRESHOLD_MODERATE = 0.10


def calculate_psi(
    reference: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculates the Population Stability Index (PSI) between reference and current feature distributions.

    PSI Formula:
        PSI = SUM [ (Actual% - Expected%) * ln(Actual% / Expected%) ]

    Thresholds:
        - PSI < 0.10: No significant drift (Stable)
        - 0.10 <= PSI < 0.25: Moderate drift (Monitor)
        - PSI >= 0.25: Significant drift (Action Required / Retrain)
    """
    ref_clean = np.asarray(reference, dtype=float)
    curr_clean = np.asarray(current, dtype=float)

    ref_clean = ref_clean[~np.isnan(ref_clean)]
    curr_clean = curr_clean[~np.isnan(curr_clean)]

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0

    # Determine quantile bins based on reference baseline
    quantiles = np.linspace(0, 100, num_buckets + 1)
    bin_edges = np.percentile(ref_clean, quantiles)
    # Deduplicate bin edges in case of low variance
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) <= 2:
        # Fallback to linear histogram bins if quantiles collapse
        bin_edges = np.linspace(
            min(ref_clean.min(), curr_clean.min()),
            max(ref_clean.max(), curr_clean.max()),
            num_buckets + 1,
        )

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    curr_counts, _ = np.histogram(curr_clean, bins=bin_edges)

    ref_pct = (ref_counts / len(ref_clean)) + epsilon
    curr_pct = (curr_counts / len(curr_clean)) + epsilon

    # Normalize to sum to 1
    ref_pct /= np.sum(ref_pct)
    curr_pct /= np.sum(curr_pct)

    psi_value = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(max(0.0, psi_value))


def compute_feature_drift_stats(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> dict:
    """Computes PSI and KS-test p-values for each feature."""
    if feature_cols is None:
        exclude = {
            "Class",
            "request_id",
            "timestamp",
            "model_version",
            "fraud_score",
            "is_fraud",
            "latency_ms",
            "ground_truth",
        }
        feature_cols = [
            c for c in reference_df.columns if c in current_df.columns and c not in exclude
        ]

    feature_metrics = {}
    drifted_features = []

    for col in feature_cols:
        ref_series = reference_df[col].dropna()
        curr_series = current_df[col].dropna()

        if len(ref_series) < 10 or len(curr_series) < 10:
            continue

        # PSI
        psi = calculate_psi(ref_series, curr_series)

        # KS Test (Kolmogorov-Smirnov)
        ks_stat, p_val = stats.ks_2samp(ref_series, curr_series)

        is_drifted = (psi >= PSI_THRESHOLD_CRITICAL) or (
            p_val < 0.01 and psi >= PSI_THRESHOLD_MODERATE
        )

        feature_metrics[col] = {
            "psi": round(psi, 4),
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_val), 6),
            "is_drifted": is_drifted,
        }

        if is_drifted:
            drifted_features.append(col)

    share_drifted = len(drifted_features) / max(1, len(feature_cols))
    max_psi = max([m["psi"] for m in feature_metrics.values()]) if feature_metrics else 0.0

    return {
        "feature_metrics": feature_metrics,
        "drifted_features": drifted_features,
        "share_drifted": round(share_drifted, 4),
        "max_feature_psi": round(max_psi, 4),
    }


def generate_evidently_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_html_path: Path | None = None,
) -> bool:
    """Attempts to generate an Evidently AI HTML Data Drift report."""
    if output_html_path is None:
        output_html_path = REPORTS_DIR / "drift_report.html"

    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        exclude = [
            "Class",
            "request_id",
            "timestamp",
            "model_version",
            "fraud_score",
            "is_fraud",
            "latency_ms",
            "ground_truth",
        ]
        cols = [c for c in reference_df.columns if c in current_df.columns and c not in exclude]

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_df[cols], current_data=current_df[cols])
        report.save_html(str(output_html_path))
        logger.info("Evidently HTML drift report generated at %s", output_html_path)
        return True
    except Exception as exc:
        logger.warning(
            "Evidently report generation skipped (%s). Using native statistical report.", exc
        )
        return False


def evaluate_drift(
    current_df: pd.DataFrame | None = None,
    reference_df: pd.DataFrame | None = None,
    save_reports: bool = True,
) -> dict:
    """Evaluates whether recent traffic exhibits significant feature/data drift compared to baseline.

    Returns:
        Drift summary dictionary with decision flag and per-feature diagnostics.
    """
    # 1. Load Reference baseline
    if reference_df is None:
        ref_path = PROCESSED_DIR / "train.parquet"
        if not ref_path.exists():
            ref_path, _, _ = split_temporal()
        reference_df = pd.read_parquet(ref_path)

    # 2. Load Current window
    if current_df is None or len(current_df) == 0:
        from serving.logger import prediction_logger

        current_df = prediction_logger.get_recent_dataframe(limit=1000)

    if len(current_df) < 20:
        logger.info(
            "Insufficient production samples (%d) to evaluate drift reliably.", len(current_df)
        )
        return {
            "drift_detected": False,
            "dataset_drift_score": 0.0,
            "max_feature_psi": 0.0,
            "drifted_features": [],
            "sample_window_size": len(current_df),
            "last_evaluated_timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": "Insufficient samples (< 20)",
        }

    # 3. Compute stats
    stats_dict = compute_feature_drift_stats(reference_df, current_df)
    share_drifted = stats_dict["share_drifted"]
    max_psi = stats_dict["max_feature_psi"]
    drifted_features = stats_dict["drifted_features"]

    # Decision rule: Trigger drift if >20% of features drifted OR max feature PSI >= 0.25
    drift_detected = (share_drifted >= 0.20) or (max_psi >= PSI_THRESHOLD_CRITICAL)

    result = {
        "drift_detected": bool(drift_detected),
        "dataset_drift_score": share_drifted,
        "max_feature_psi": max_psi,
        "drifted_features": drifted_features,
        "sample_window_size": len(current_df),
        "last_evaluated_timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_metrics": stats_dict["feature_metrics"],
    }

    # 4. Save reports
    if save_reports:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = REPORTS_DIR / "drift_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        generate_evidently_report(reference_df, current_df, REPORTS_DIR / "drift_report.html")

    logger.info(
        "Drift Assessment Result: DriftDetected=%s (ShareDrifted=%.2f%%, MaxPSI=%.4f, DriftedFeatures=%s)",
        drift_detected,
        share_drifted * 100,
        max_psi,
        drifted_features[:5],
    )

    return result


def get_latest_drift_summary() -> dict:
    """Loads the latest cached drift summary or evaluates on current logs."""
    summary_path = REPORTS_DIR / "drift_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return evaluate_drift(save_reports=False)


if __name__ == "__main__":
    evaluate_drift()
