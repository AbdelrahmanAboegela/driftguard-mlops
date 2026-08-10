"""Imbalance-handling utilities applied only to model training data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE

SUPPORTED_RESAMPLING_METHODS = frozenset({"none", "smote", "adasyn", "smoteenn", "smotetomek"})


def resample_training_data(
    features: pd.DataFrame,
    labels: np.ndarray,
    method: str = "none",
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Resample a training fold without ever touching validation or production data.

    ``none`` retains the original class distribution for cost-sensitive learning.
    All other methods balance the minority class to the majority class.
    """
    normalized_method = method.lower().strip()
    if normalized_method not in SUPPORTED_RESAMPLING_METHODS:
        supported = ", ".join(sorted(SUPPORTED_RESAMPLING_METHODS))
        raise ValueError(f"Unsupported resampling method '{method}'. Choose one of: {supported}.")

    y = np.asarray(labels, dtype=int)
    if normalized_method == "none":
        return features.reset_index(drop=True), y

    class_counts = np.bincount(y)
    minority_count = int(class_counts.min()) if len(class_counts) == 2 else 0
    if minority_count < 2:
        raise ValueError(
            "Resampling requires at least two examples from each class in the training fold."
        )

    neighbors = min(5, minority_count - 1)
    if normalized_method == "smote":
        sampler = SMOTE(random_state=random_state, k_neighbors=neighbors)
    elif normalized_method == "adasyn":
        sampler = ADASYN(random_state=random_state, n_neighbors=neighbors)
    elif normalized_method == "smoteenn":
        sampler = SMOTEENN(
            random_state=random_state,
            smote=SMOTE(random_state=random_state, k_neighbors=neighbors),
        )
    else:
        sampler = SMOTETomek(
            random_state=random_state,
            smote=SMOTE(random_state=random_state, k_neighbors=neighbors),
        )

    resampled_features, resampled_labels = sampler.fit_resample(features, y)
    return pd.DataFrame(resampled_features, columns=features.columns), np.asarray(resampled_labels)
