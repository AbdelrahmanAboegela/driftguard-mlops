"""Temporal dataset splitter for DriftGuard.

Splits data temporally (sorted by Time):
- 70% Historical training set
- 15% Initial holdout / validation set (frozen benchmark for champion-challenger)
- 15% Production replay stream (held back for drift injection & replay simulation)
"""

from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
from data.get_data import ensure_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parent / "processed"


def split_temporal(
    csv_path: Path | None = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
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

    n_train = int(n_total * train_ratio)
    n_val = int(n_total * (train_ratio + val_ratio))

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_val].copy()
    prod_stream_df = df.iloc[n_val:].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val_holdout.parquet"
    prod_path = output_dir / "prod_stream.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    prod_stream_df.to_parquet(prod_path, index=False)

    logger.info(
        "Temporal Split Complete:\n"
        "  - Training (70%%): %d samples (%d frauds, %.3f%%)\n"
        "  - Holdout  (15%%): %d samples (%d frauds, %.3f%%)\n"
        "  - Prod Stream (15%%): %d samples (%d frauds, %.3f%%)",
        len(train_df),
        train_df["Class"].sum(),
        train_df["Class"].mean() * 100,
        len(val_df),
        val_df["Class"].sum(),
        val_df["Class"].mean() * 100,
        len(prod_stream_df),
        prod_stream_df["Class"].sum(),
        prod_stream_df["Class"].mean() * 100,
    )

    return train_path, val_path, prod_path


if __name__ == "__main__":
    split_temporal()
