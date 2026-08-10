"""Metrics exporter and drift evaluation runner for DriftGuard."""

from __future__ import annotations

import argparse
import logging
import time

from monitoring.drift_detector import evaluate_drift

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_drift_monitor_iteration() -> dict:
    """Executes a single drift evaluation cycle and outputs key metrics."""
    logger.info("Starting drift assessment cycle...")
    result = evaluate_drift(save_reports=True)

    status_str = (
        "[!] DRIFT ALERT DETECTED" if result["drift_detected"] else "[OK] SYSTEM HEALTHY (No Drift)"
    )
    logger.info(
        "\n================ DRIFTGUARD STATUS ================\n"
        "  Status:             %s\n"
        "  Dataset Drift Rate: %.2f%%\n"
        "  Max Feature PSI:    %.4f\n"
        "  Drifted Features:   %s\n"
        "  Samples Evaluated:  %d\n"
        "===================================================",
        status_str,
        result["dataset_drift_score"] * 100,
        result["max_feature_psi"],
        result["drifted_features"] or "None",
        result["sample_window_size"],
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="DriftGuard Drift Monitor & Exporter")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    if args.once:
        run_drift_monitor_iteration()
        return

    logger.info("Starting continuous drift monitor loop (interval=%ds)...", args.interval)
    while True:
        try:
            run_drift_monitor_iteration()
        except Exception as exc:
            logger.error("Error in drift monitoring iteration: %s", exc)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
