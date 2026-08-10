"""Benchmark supported imbalance strategies on a fixed temporal test period."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import xgboost as xgb

from data.get_data import generate_synthetic_fraud_dataset
from training.evaluate import bootstrap_confidence_intervals, evaluate_predictions
from training.feature_engineering import FeatureTransformer
from training.imbalance import SUPPORTED_RESAMPLING_METHODS, resample_training_data

REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "imbalance_benchmark.csv"
METHODS = ("none", "smote", "adasyn", "smoteenn", "smotetomek")


def benchmark(
    n_samples: int = 120_000,
    random_state: int = 42,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 25.0,
) -> pd.DataFrame:
    """Run every supported resampling configuration with an untouched temporal test set."""
    if set(METHODS) - SUPPORTED_RESAMPLING_METHODS:
        raise RuntimeError("Benchmark methods must be supported by the imbalance module.")

    data = generate_synthetic_fraud_dataset(n_samples=n_samples, random_seed=random_state)
    train_end = int(len(data) * 0.65)
    validation_end = int(len(data) * 0.80)
    production_end = int(len(data) * 0.90)
    train_df = data.iloc[:train_end]
    validation_df = data.iloc[train_end:validation_end]
    test_df = data.iloc[production_end:]

    transformer = FeatureTransformer()
    train_features = transformer.fit(train_df).transform(train_df)
    validation_features = transformer.transform(validation_df)
    test_features = transformer.transform(test_df)
    train_labels = train_df["Class"].to_numpy()
    validation_labels = validation_df["Class"].to_numpy()
    test_labels = test_df["Class"].to_numpy()

    results = []
    for method in METHODS:
        started_at = time.perf_counter()
        sampled_features, sampled_labels = resample_training_data(
            train_features, train_labels, method=method, random_state=random_state
        )
        negatives = int((sampled_labels == 0).sum())
        positives = int((sampled_labels == 1).sum())
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=negatives / max(1, positives) if method == "none" else 1.0,
            eval_metric="aucpr",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )
        model.fit(sampled_features, sampled_labels, verbose=False)

        validation_probs = model.predict_proba(validation_features)[:, 1]
        validation_metrics = evaluate_predictions(
            validation_labels,
            validation_probs,
            threshold_strategy="cost",
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
        )
        test_probs = model.predict_proba(test_features)[:, 1]
        test_metrics = evaluate_predictions(
            test_labels,
            test_probs,
            threshold=validation_metrics["threshold"],
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
        )
        intervals = bootstrap_confidence_intervals(
            test_labels,
            test_probs,
            threshold=validation_metrics["threshold"],
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
        )
        results.append(
            {
                "method": method,
                "train_rows": len(sampled_features),
                "test_rows": len(test_df),
                "test_fraud_cases": int(test_labels.sum()),
                "threshold": validation_metrics["threshold"],
                "pr_auc": test_metrics["pr_auc"],
                "f1": test_metrics["f1"],
                "recall": test_metrics["recall"],
                "specificity": test_metrics["specificity"],
                "g_mean": test_metrics["g_mean"],
                "false_negatives": test_metrics["false_negatives"],
                "false_positives": test_metrics["false_positives"],
                "expected_cost": test_metrics["expected_cost"],
                "pr_auc_ci_lower": intervals["pr_auc"]["lower"],
                "pr_auc_ci_upper": intervals["pr_auc"]["upper"],
                "g_mean_ci_lower": intervals["g_mean"]["lower"],
                "g_mean_ci_upper": intervals["g_mean"]["upper"],
                "runtime_seconds": time.perf_counter() - started_at,
            }
        )

    report = pd.DataFrame(results).sort_values(["expected_cost", "pr_auc"], ascending=[True, False])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark all supported fraud-imbalance strategies."
    )
    parser.add_argument("--samples", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--false-positive-cost", type=float, default=1.0)
    parser.add_argument("--false-negative-cost", type=float, default=25.0)
    args = parser.parse_args()
    report = benchmark(
        n_samples=args.samples,
        random_state=args.seed,
        false_positive_cost=args.false_positive_cost,
        false_negative_cost=args.false_negative_cost,
    )
    print(report.to_string(index=False))
    print(f"\nSaved benchmark results to {REPORT_PATH}")


if __name__ == "__main__":
    main()
