"""Synthetic drift injection module for DriftGuard production simulation and testing."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def inject_feature_drift(
    df: pd.DataFrame,
    amount_scale: float = 2.2,
    pca_shift_features: list[str] | None = None,
    pca_shift_std: float = 1.8,
    random_seed: int = 101,
) -> pd.DataFrame:
    """Simulates realistic covariate data drift.

    Modifications:
    1. Inflation / merchant pricing shift: Multiplies Amount by amount_scale with non-linear heavy tails.
    2. Covariate shift: Perturbs PCA components (e.g. V1, V2, V4, V11, V12) simulating new payment channels and cross-border gateways.
    """
    df_drifted = df.reset_index(drop=True).copy()
    rng = np.random.default_rng(random_seed)
    n = len(df_drifted)

    # 1. Amount Drift (Macroeconomic inflation + large holiday shopping transactions)
    amount_jitter = rng.lognormal(mean=0.3, sigma=0.4, size=n)
    df_drifted["Amount"] = np.clip(
        df_drifted["Amount"] * amount_scale * amount_jitter, 0.5, 35000.0
    )

    # 2. PCA Covariate Drift
    if pca_shift_features is None:
        pca_shift_features = ["V1", "V2", "V3", "V4", "V10", "V11", "V12", "V14", "V17"]

    for col in pca_shift_features:
        if col in df_drifted.columns:
            # Add systematic directional shift plus noise
            shift_mean = rng.uniform(1.2, 2.5) * rng.choice([-1, 1])
            noise = rng.normal(loc=shift_mean, scale=pca_shift_std, size=n)
            df_drifted[col] = df_drifted[col] + noise

    logger.info(
        "Injected feature drift: Amount scaled by %.2fx, %d PCA features shifted.",
        amount_scale,
        len(pca_shift_features),
    )
    return df_drifted


def inject_concept_drift(
    df: pd.DataFrame,
    new_fraud_ratio: float = 0.025,
    random_seed: int = 202,
) -> pd.DataFrame:
    """Simulates concept drift (adversarial fraud pattern evolution).

    Fraudsters adopt micro-transaction testing (low Amount, high velocity)
    and alter their behavior to bypass historical decision boundaries.
    """
    df_drifted = df.reset_index(drop=True).copy()
    rng = np.random.default_rng(random_seed)
    n = len(df_drifted)

    # Boost fraud occurrence (fraud wave)
    n_extra_fraud = int(n * new_fraud_ratio)
    fraud_indices = rng.choice(n, size=n_extra_fraud, replace=False)

    df_drifted.loc[fraud_indices, "Class"] = 1
    # For new evasion fraud, set subtle amounts ($1.00 - $15.00 card testing)
    df_drifted.loc[fraud_indices, "Amount"] = np.round(
        rng.uniform(1.0, 15.0, size=n_extra_fraud), 2
    )

    # Evasion patterns in latent space
    for col, shift in [("V4", 2.5), ("V11", 2.0), ("V14", -3.5)]:
        if col in df_drifted.columns:
            df_drifted.loc[fraud_indices, col] += rng.normal(shift, 0.8, size=n_extra_fraud)

    logger.info(
        "Injected concept drift: elevated fraud rate to %.2f%% with evasion patterns.",
        df_drifted["Class"].mean() * 100,
    )
    return df_drifted


def create_drifted_production_stream(
    prod_stream_df: pd.DataFrame,
    drift_onset_fraction: float = 0.35,
    amount_scale: float = 2.0,
    concept_drift: bool = True,
) -> tuple[pd.DataFrame, int]:
    """Constructs a realistic production replay dataset with a clean pre-drift phase and a drifted post-drift phase.

    Returns:
        (stream_df, drift_onset_index)
    """
    n_total = len(prod_stream_df)
    n_clean = int(n_total * drift_onset_fraction)

    clean_segment = prod_stream_df.iloc[:n_clean].copy()
    post_drift_segment = prod_stream_df.iloc[n_clean:].copy()

    # Apply drift to post-drift segment
    drifted_segment = inject_feature_drift(post_drift_segment, amount_scale=amount_scale)
    if concept_drift:
        drifted_segment = inject_concept_drift(drifted_segment, new_fraud_ratio=0.02)

    combined_df = pd.concat([clean_segment, drifted_segment], ignore_index=True)
    logger.info(
        "Created production replay stream: %d clean requests, %d drifted requests (onset at step %d).",
        len(clean_segment),
        len(drifted_segment),
        n_clean,
    )
    return combined_df, n_clean
