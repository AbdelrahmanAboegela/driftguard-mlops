"""Temporal dataset splitter for DriftGuard.

Splits data temporally (sorted by Time):
- 65% Historical training set
- 15% Initial holdout / validation set (frozen benchmark for champion-challenger)
- 10% Production replay stream (held back for drift injection & replay simulation)
- 10% Final test set, never used for training, threshold selection, or model promotion
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from data.get_data import ensure_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parent / "processed"
TEST_HOLDOUT_PATH = PROCESSED_DIR / "test_holdout.parquet"


def split_temporal(
    csv_path: Path | None = None,
    train_ratio: float = 0.65,
    val_ratio: float = 0.15,
    production_ratio: float = 0.10,
    output_dir: Path = PROCESSED_DIR,
) -> tuple[Path, Path, Path]:
    """Splits dataset strictly by Time to prevent temporal data leakage."""
    if csv_path is None or not csv_path.exists():
        csv_path = ensure_dataset()

    logger.info("Loading raw dataset from %s...", csv_path)
    df = pd.read_csv(csv_path)

    # Sort strictly by Time
    df = df.sort_values(by="Time").reset_index(drop=True)
    n_total = len(df)

    if train_ratio <= 0 or val_ratio <= 0 or production_ratio <= 0:
        raise ValueError("All temporal split ratios must be positive.")
    if train_ratio + val_ratio + production_ratio >= 1:
        raise ValueError(
            "Train, validation, and production ratios must leave a positive test holdout."
        )

    n_train = int(n_total * train_ratio)
    n_val = int(n_total * (train_ratio + val_ratio))
    n_production = int(n_total * (train_ratio + val_ratio + production_ratio))

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_val].copy()
    prod_stream_df = df.iloc[n_val:n_production].copy()
    test_df = df.iloc[n_production:].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val_holdout.parquet"
    prod_path = output_dir / "prod_stream.parquet"
    test_path = output_dir / "test_holdout.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    prod_stream_df.to_parquet(prod_path, index=False)
    test_df.to_parquet(test_path, index=False)

    logger.info(
        "Temporal Split Complete:\n"
        "  - Training (65%%): %d samples (%d frauds, %.3f%%)\n"
        "  - Holdout  (15%%): %d samples (%d frauds, %.3f%%)\n"
        "  - Prod Stream (10%%): %d samples (%d frauds, %.3f%%)\n"
        "  - Test Holdout (10%%): %d samples (%d frauds, %.3f%%)",
        len(train_df),
        train_df["Class"].sum(),
        train_df["Class"].mean() * 100,
        len(val_df),
        val_df["Class"].sum(),
        val_df["Class"].mean() * 100,
        len(prod_stream_df),
        prod_stream_df["Class"].sum(),
        prod_stream_df["Class"].mean() * 100,
        len(test_df),
        test_df["Class"].sum(),
        test_df["Class"].mean() * 100,
    )

    return train_path, val_path, prod_path


if __name__ == "__main__":
    split_temporal()
