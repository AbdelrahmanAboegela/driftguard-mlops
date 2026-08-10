"""Evaluation metrics and champion-challenger validation module for DriftGuard."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> tuple[float, float]:
    """Finds the decision threshold that maximizes F1 score on validation data.

    Returns:
        (best_threshold, best_f1)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    # Avoid division by zero
    denom = precisions + recalls
    f1_scores = np.where(denom > 0, (2 * precisions * recalls) / np.maximum(denom, 1e-9), 0.0)

    # precision_recall_curve thresholds has len = len(precisions) - 1
    if len(thresholds) == 0:
        return 0.5, 0.0

    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    return best_threshold, best_f1


def find_cost_optimal_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 25.0,
) -> tuple[float, float]:
    """Finds the threshold with the minimum expected classification cost."""
    if false_positive_cost < 0 or false_negative_cost < 0:
        raise ValueError("Misclassification costs must be non-negative.")

    thresholds = np.unique(np.asarray(y_probs, dtype=float))
    if len(thresholds) == 0:
        return 0.5, 0.0

    y_true = np.asarray(y_true, dtype=int)
    costs = []
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        false_positives = int(((y_pred == 1) & (y_true == 0)).sum())
        false_negatives = int(((y_pred == 0) & (y_true == 1)).sum())
        costs.append(false_positives * false_positive_cost + false_negatives * false_negative_cost)

    best_index = int(np.argmin(costs))
    return float(thresholds[best_index]), float(costs[best_index])


def evaluate_predictions(
    y_true: np.ndarray | list,
    y_probs: np.ndarray | list,
    threshold: float | None = None,
    threshold_strategy: str = "f1",
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 25.0,
) -> dict[str, float]:
    """Calculates comprehensive fraud detection metrics.

    Focuses heavily on PR-AUC (Average Precision) and F1 rather than ROC-AUC,
    because on highly imbalanced fraud datasets (0.17% positives), ROC-AUC gives
    an overly optimistic score due to the large volume of true negatives.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)

    if threshold is None:
        if threshold_strategy == "f1":
            threshold, _ = find_optimal_threshold(y_true, y_probs)
        elif threshold_strategy == "cost":
            threshold, _ = find_cost_optimal_threshold(
                y_true,
                y_probs,
                false_positive_cost=false_positive_cost,
                false_negative_cost=false_negative_cost,
            )
        else:
            raise ValueError("threshold_strategy must be either 'f1' or 'cost'.")

    y_pred = (y_probs >= threshold).astype(int)

    # Primary imbalanced metrics
    pr_auc = float(average_precision_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else 0.5
    brier = float(brier_score_loss(y_true, y_probs))

    # Confusion matrix breakdowns
    if len(np.unique(y_true)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        g_mean = float(np.sqrt(recall * specificity))
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        fnr, fpr, specificity, g_mean = 0.0, 0.0, 0.0, 0.0

    expected_cost = float(fp * false_positive_cost + fn * false_negative_cost)

    metrics = {
        "pr_auc": pr_auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "false_negative_rate": fnr,
        "false_positive_rate": fpr,
        "specificity": specificity,
        "g_mean": g_mean,
        "expected_cost": expected_cost,
        "cost_per_transaction": expected_cost / max(1, len(y_true)),
        "threshold": float(threshold),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }

    logger.info(
        "Evaluation Results (Threshold=%.4f):\n"
        "  - PR-AUC:    %.4f\n"
        "  - F1 Score:  %.4f\n"
        "  - Precision: %.4f\n"
        "  - Recall:    %.4f\n"
        "  - ROC-AUC:   %.4f\n"
        "  - FNR (Missed Fraud): %.2f%%",
        threshold,
        pr_auc,
        f1,
        precision,
        recall,
        roc_auc,
        fnr * 100,
    )

    return metrics


def bootstrap_confidence_intervals(
    y_true: np.ndarray | list,
    y_probs: np.ndarray | list,
    threshold: float,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 25.0,
    iterations: int = 200,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """Returns percentile bootstrap confidence intervals for a fixed test threshold."""
    if iterations < 20:
        raise ValueError("At least 20 bootstrap iterations are required.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one.")

    labels = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(y_probs, dtype=float)
    if len(labels) != len(probabilities) or len(labels) == 0:
        raise ValueError("y_true and y_probs must be non-empty arrays of equal length.")

    rng = np.random.default_rng(random_state)
    metric_names = ("pr_auc", "f1", "recall", "specificity", "g_mean", "expected_cost")
    samples = {name: [] for name in metric_names}
    for _ in range(iterations):
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_metrics = evaluate_predictions(
            labels[indices],
            probabilities[indices],
            threshold=threshold,
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
        )
        for name in metric_names:
            samples[name].append(sampled_metrics[name])

    alpha = (1 - confidence_level) / 2
    return {
        name: {
            "lower": float(np.quantile(values, alpha)),
            "upper": float(np.quantile(values, 1 - alpha)),
        }
        for name, values in samples.items()
    }
