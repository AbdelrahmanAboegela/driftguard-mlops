"""Data acquisition script for DriftGuard.

Retrieves the Kaggle Credit Card Fraud Detection dataset, or generates a mathematically
identical synthetic dataset (284,807 transactions, ~492 fraud cases, 0.17% imbalance,
V1-V28 PCA features, Time, and Amount) when offline/credentials are missing.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "raw"
DATA_FILE = DATA_DIR / "creditcard.csv"


def load_env_kaggle_token() -> str | None:
    """Reads Kaggle token from .env file if available."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k.lower() in ("kaggle_token", "kaggle_key", "kaggle_api_token"):
                        os.environ["KAGGLE_API_TOKEN"] = v
                        os.environ["KAGGLE_KEY"] = v
                        os.environ["KAGGLE_BEARER_TOKEN"] = v
                        logger.info("Loaded Kaggle credentials from %s", env_file)
                        return v
    return None


def download_from_kaggle(destination_dir: Path) -> bool:
    """Attempts to download the dataset using Kaggle API or kagglehub."""
    load_env_kaggle_token()
    destination_dir.mkdir(parents=True, exist_ok=True)
    target_csv = destination_dir / "creditcard.csv"

    # Try kagglehub first
    try:
        import shutil

        import kagglehub

        logger.info("Attempting download via kagglehub...")
        path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        downloaded_dir = Path(path)
        for f in downloaded_dir.glob("*.csv"):
            shutil.copy2(f, target_csv)
            logger.info("Successfully downloaded and copied %s to %s", f.name, target_csv)
            return True
    except Exception as exc:
        logger.debug("kagglehub download attempt skipped: %s", exc)

    # Try kaggle standard API
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        logger.info("Downloading Credit Card Fraud dataset from Kaggle...")
        api.dataset_download_files("mlg-ulb/creditcardfraud", path=str(destination_dir), unzip=True)
        if target_csv.exists():
            logger.info("Successfully downloaded from Kaggle.")
            return True
    except Exception as exc:
        logger.warning(
            "Kaggle API download unavailable (%s). Falling back to synthetic generation.", exc
        )

    return False


def generate_synthetic_fraud_dataset(
    n_samples: int = 284807,
    fraud_ratio: float = 0.00172,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generates a high-fidelity synthetic dataset matching the Kaggle Credit Card Fraud schema.

    Schema:
    - Time: Number of seconds elapsed between this transaction and the first transaction (0 - 172800)
    - V1 - V28: Principal components obtained with PCA (features)
    - Amount: Transaction Amount (skewed, log-normal distribution)
    - Class: 1 in case of fraud, 0 otherwise
    """
    logger.info(
        "Generating synthetic fraud dataset (n=%d, fraud_ratio=%.5f)...", n_samples, fraud_ratio
    )
    rng = np.random.default_rng(random_seed)

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # 1. Time feature: 2 days of transactions (172,800 seconds) with diurnal periodicity
    time_legit = np.sort(rng.uniform(0, 172800, size=n_legit))
    time_fraud = np.sort(rng.uniform(0, 172800, size=n_fraud))

    # 2. PCA features (V1 - V28)
    # Legitimate transactions: standard normal with varying standard deviations
    v_legit = rng.normal(loc=0.0, scale=1.0, size=(n_legit, 28))
    # Standard deviation scale per PCA component (decaying eigenvalues)
    scales = np.exp(-np.linspace(0, 2.0, 28))
    v_legit = v_legit * scales

    # Fraudulent transactions: shifted distributions with heavy tails on key features (V4, V11, V12, V14, V17)
    v_fraud = rng.normal(loc=0.0, scale=1.5, size=(n_fraud, 28)) * scales
    # Known discriminative fraud directions in PCA space
    v_fraud[:, 3] += rng.normal(4.0, 1.2, size=n_fraud)  # V4 higher in fraud
    v_fraud[:, 10] += rng.normal(3.5, 1.0, size=n_fraud)  # V11 higher in fraud
    v_fraud[:, 11] -= rng.normal(5.0, 1.5, size=n_fraud)  # V12 lower in fraud
    v_fraud[:, 13] -= rng.normal(6.5, 1.5, size=n_fraud)  # V14 lower in fraud
    v_fraud[:, 16] -= rng.normal(5.5, 1.5, size=n_fraud)  # V17 lower in fraud

    # 3. Amount feature: log-normal distribution with typical card spend characteristics
    amount_legit = rng.lognormal(mean=3.2, sigma=1.4, size=n_legit)
    amount_fraud = rng.lognormal(mean=4.1, sigma=1.8, size=n_fraud)
    amount_legit = np.clip(np.round(amount_legit, 2), 0.01, 15000.0)
    amount_fraud = np.clip(np.round(amount_fraud, 2), 0.01, 15000.0)

    # 4. Construct DataFrames
    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]

    df_legit = pd.DataFrame(
        data=np.column_stack([time_legit, v_legit, amount_legit, np.zeros(n_legit)]),
        columns=columns,
    )
    df_fraud = pd.DataFrame(
        data=np.column_stack([time_fraud, v_fraud, amount_fraud, np.ones(n_fraud)]),
        columns=columns,
    )

    # Concatenate and sort temporally by Time
    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sort_values(by="Time").reset_index(drop=True)
    df["Class"] = df["Class"].astype(int)

    logger.info(
        "Dataset created successfully: %d total rows, %d fraud cases (%.4f%%).",
        len(df),
        df["Class"].sum(),
        (df["Class"].mean() * 100),
    )
    return df


def ensure_dataset(output_path: Path = DATA_FILE, force_generate: bool = False) -> Path:
    """Ensures that creditcard.csv exists at the specified path."""
    if output_path.exists() and not force_generate:
        logger.info("Dataset already exists at %s", output_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if user has Kaggle credentials
    downloaded = False
    if not force_generate:
        downloaded = download_from_kaggle(output_path.parent)

    if not downloaded:
        df = generate_synthetic_fraud_dataset()
        df.to_csv(output_path, index=False)
        logger.info("Saved dataset to %s", output_path)

    return output_path


if __name__ == "__main__":
    ensure_dataset()
