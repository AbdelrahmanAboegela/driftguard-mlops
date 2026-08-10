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


def evaluate_predictions(
    y_true: np.ndarray | list,
    y_probs: np.ndarray | list,
    threshold: float | None = None,
) -> dict[str, float]:
    """Calculates comprehensive fraud detection metrics.

    Focuses heavily on PR-AUC (Average Precision) and F1 rather than ROC-AUC,
    because on highly imbalanced fraud datasets (0.17% positives), ROC-AUC gives
    an overly optimistic score due to the large volume of true negatives.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)

    if threshold is None:
        best_thresh, _ = find_optimal_threshold(y_true, y_probs)
        threshold = best_thresh

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
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        fnr, fpr = 0.0, 0.0

    metrics = {
        "pr_auc": pr_auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "false_negative_rate": fnr,
        "false_positive_rate": fpr,
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
