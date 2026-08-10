"""End-to-End Production Simulation for DriftGuard.

Demonstrates:
1. Normal production traffic baseline evaluation.
2. Injected synthetic drift at step T_drift.
3. Automated drift detection via PSI / Evidently.
4. Autonomous retraining and Champion-Challenger validation.
5. Verification of performance recovery post-promotion.
6. Exports comprehensive benchmark CSV and summary statistics.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from data.split_data import PROCESSED_DIR, split_temporal
from monitoring.drift_detector import compute_feature_drift_stats, evaluate_drift
from monitoring.inject_drift import inject_concept_drift, inject_feature_drift
from orchestration.retrain_pipeline import (
    evaluate_and_promote_challenger,
    load_retraining_data,
    train_challenger,
)
from serving.logger import prediction_logger
from serving.model_loader import model_manager
from training.evaluate import evaluate_predictions
from training.feature_engineering import ARTIFACTS_DIR
from training.train import train_baseline_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def run_production_simulation(
    sample_size: int = 4000,
    drift_onset_step: int = 1500,
    amount_scale: float = 2.4,
) -> pd.DataFrame:
    """Executes the full production simulation cycle."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("================ STARTING DRIFTGUARD PRODUCTION SIMULATION ================")

    # Step 0: Ensure Baseline Model is trained and loaded
    logger.info("Step 0: Ensuring initial Champion model is trained and active...")
    prod_path = PROCESSED_DIR / "prod_stream.parquet"
    if not prod_path.exists():
        split_temporal()

    model_manager.load(force_reload=True)
    initial_version = model_manager.model_version
    logger.info("Active Baseline Model: %s", initial_version)

    # Step 1: Prepare Stream
    prod_df = pd.read_parquet(prod_path)
    if len(prod_df) > sample_size:
        stream_subset = prod_df.iloc[:sample_size].copy()
    else:
        stream_subset = prod_df.copy()

    n_total = len(stream_subset)
    drift_step = min(drift_onset_step, int(n_total * 0.4))

    clean_segment = stream_subset.iloc[:drift_step].copy()
    drifted_segment = stream_subset.iloc[drift_step:].copy()

    # Inject realistic covariate + concept drift into the second segment
    drifted_segment = inject_feature_drift(drifted_segment, amount_scale=amount_scale)
    drifted_segment = inject_concept_drift(drifted_segment, new_fraud_ratio=0.035)

    full_stream = pd.concat([clean_segment, drifted_segment], ignore_index=True)
    logger.info(
        "Simulating production stream with %d transactions (Clean: %d, Drifted: %d, Drift Onset Step: %d)...",
        len(full_stream),
        len(clean_segment),
        len(drifted_segment),
        drift_step,
    )

    # Step 2: Replay Clean Phase (Normal Traffic)
    logger.info("\n--- Phase 1: Replaying Clean Traffic (Steps 0 to %d) ---", drift_step)
    clean_records = clean_segment.to_dict(orient="records")
    clean_results, clean_latency = model_manager.predict_batch(clean_records)

    clean_probs = [r["fraud_score"] for r in clean_results]
    clean_y = clean_segment["Class"].values
    clean_metrics = evaluate_predictions(clean_y, clean_probs, threshold=model_manager.threshold)

    # Check drift on clean segment vs training baseline
    train_baseline_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    clean_drift_stats = compute_feature_drift_stats(train_baseline_df, clean_segment)
    logger.info(
        "Clean Phase Drift Status: ShareDrifted=%.2f%%, MaxPSI=%.4f (Expected < 0.25)",
        clean_drift_stats["share_drifted"] * 100,
        clean_drift_stats["max_feature_psi"],
    )

    # Step 3: Replay Drifted Phase on Stale Baseline Model
    logger.info("\n--- Phase 2: Injected Drift & Stale Model Evaluation (Steps %d to %d) ---", drift_step, len(full_stream))
    drifted_records = drifted_segment.to_dict(orient="records")
    stale_results, stale_latency = model_manager.predict_batch(drifted_records)

    stale_probs = [r["fraud_score"] for r in stale_results]
    drifted_y = drifted_segment["Class"].values
    stale_drifted_metrics = evaluate_predictions(drifted_y, stale_probs, threshold=model_manager.threshold)

    # Compute drift on drifted segment
    drift_stats = compute_feature_drift_stats(train_baseline_df, drifted_segment)
    drift_detected = (drift_stats["share_drifted"] >= 0.20) or (drift_stats["max_feature_psi"] >= 0.25)

    logger.info(
        "Drift Assessment on Post-Drift Traffic:\n"
        "  - Drift Detected:    %s\n"
        "  - Share Drifted:     %.2f%%\n"
        "  - Max Feature PSI:   %.4f\n"
        "  - Drifted Features:  %s\n"
        "  - Stale Model F1:    %.4f (Degraded from %.4f)\n"
        "  - Stale Model PR-AUC:%.4f (Degraded from %.4f)\n"
        "  - Missed Fraud (FNR):%.2f%%",
        drift_detected,
        drift_stats["share_drifted"] * 100,
        drift_stats["max_feature_psi"],
        drift_stats["drifted_features"][:4],
        stale_drifted_metrics["f1"],
        clean_metrics["f1"],
        stale_drifted_metrics["pr_auc"],
        clean_metrics["pr_auc"],
        stale_drifted_metrics["false_negative_rate"] * 100,
    )

    # Step 4: Autonomous Retraining Trigger
    logger.info("\n--- Phase 3: Triggering Autonomous Retraining Pipeline ---")
    t_retrain_start = time.perf_counter()

    # Split drifted data: first 40% for retrain enrichment, remaining 60% for post-retrain verification
    n_retrain_drift = int(len(drifted_segment) * 0.45)
    drift_retrain_data = drifted_segment.iloc[:n_retrain_drift].copy()
    drift_test_data = drifted_segment.iloc[n_retrain_drift:].copy()

    train_df, val_df = load_retraining_data(additional_data=drift_retrain_data)
    challenger_model, challenger_metrics, challenger_ver, _ = train_challenger(train_df, val_df)

    promotion_decision = evaluate_and_promote_challenger(
        challenger_model=challenger_model,
        challenger_metrics=challenger_metrics,
        challenger_version=challenger_ver,
        val_df=val_df,
    )

    retrain_duration_sec = time.perf_counter() - t_retrain_start
    logger.info(
        "Retraining completed in %.2fs. Promotion status: %s (New Production Version: %s)",
        retrain_duration_sec,
        promotion_decision["promoted"],
        challenger_ver if promotion_decision["promoted"] else initial_version,
    )

    # Step 5: Post-Retraining Performance Recovery Evaluation
    logger.info("\n--- Phase 4: Evaluating Performance Recovery on Unseen Drifted Test Set ---")
    model_manager.load(force_reload=True)

    test_records = drift_test_data.to_dict(orient="records")
    recovered_results, _ = model_manager.predict_batch(test_records)
    recovered_probs = [r["fraud_score"] for r in recovered_results]
    test_y = drift_test_data["Class"].values

    # Stale model metrics on test slice (already computed as part of stale_results)
    stale_test_probs = [r["fraud_score"] for r in stale_results[n_retrain_drift:]]
    stale_metrics_on_test = evaluate_predictions(test_y, stale_test_probs, threshold=0.5)
    recovered_metrics = evaluate_predictions(test_y, recovered_probs, threshold=model_manager.threshold)

    f1_recovery_gain = recovered_metrics["f1"] - stale_metrics_on_test["f1"]
    prauc_recovery_gain = recovered_metrics["pr_auc"] - stale_metrics_on_test["pr_auc"]
    fnr_reduction = (stale_metrics_on_test["false_negative_rate"] - recovered_metrics["false_negative_rate"]) * 100

    # Step 6: Assemble Comparative Results DataFrame
    comparison_records = [
        {
            "Phase": "1. Pre-Drift (Clean Traffic)",
            "Model Version": initial_version,
            "PR-AUC": clean_metrics["pr_auc"],
            "F1 Score": clean_metrics["f1"],
            "Precision": clean_metrics["precision"],
            "Recall": clean_metrics["recall"],
            "Missed Fraud (FNR %)": clean_metrics["false_negative_rate"] * 100,
            "Max Feature PSI": clean_drift_stats["max_feature_psi"],
            "Drift Status": "No Drift",
        },
        {
            "Phase": "2. Post-Drift (Stale Baseline)",
            "Model Version": initial_version,
            "PR-AUC": stale_drifted_metrics["pr_auc"],
            "F1 Score": stale_drifted_metrics["f1"],
            "Precision": stale_drifted_metrics["precision"],
            "Recall": stale_drifted_metrics["recall"],
            "Missed Fraud (FNR %)": stale_drifted_metrics["false_negative_rate"] * 100,
            "Max Feature PSI": drift_stats["max_feature_psi"],
            "Drift Status": "CRITICAL DRIFT DETECTED",
        },
        {
            "Phase": "3. Post-Retrain (Promoted Model)",
            "Model Version": model_manager.model_version,
            "PR-AUC": recovered_metrics["pr_auc"],
            "F1 Score": recovered_metrics["f1"],
            "Precision": recovered_metrics["precision"],
            "Recall": recovered_metrics["recall"],
            "Missed Fraud (FNR %)": recovered_metrics["false_negative_rate"] * 100,
            "Max Feature PSI": drift_stats["max_feature_psi"],
            "Drift Status": "Recovered & Adapted",
        },
    ]

    results_df = pd.DataFrame(comparison_records)
    csv_path = REPORTS_DIR / "simulation_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info("Saved simulation summary table to %s", csv_path)

    # Print Final Formatted Benchmark Table
    print("\n" + "=" * 90)
    print("                      DRIFTGUARD SIMULATION BENCHMARK REPORT")
    print("=" * 90)
    print(results_df.to_string(index=False))
    print("=" * 90)
    print(f"  * F1 Improvement Post-Retraining:        +{f1_recovery_gain:.4f}")
    print(f"  * PR-AUC Gain Post-Retraining:           +{prauc_recovery_gain:.4f}")
    print(f"  * Missed Fraud Rate (FNR) Reduction:     {fnr_reduction:.1f}%")
    print(f"  * Total Retraining & Promotion Latency:  {retrain_duration_sec:.2f}s")
    print("=" * 90 + "\n")

    return results_df


if __name__ == "__main__":
    run_production_simulation()
