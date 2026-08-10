"""Feature engineering pipeline for DriftGuard fraud detection."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
FEATURE_NAMES = [
    "scaled_amount",
    "log_amount",
    "hour_sin",
    "hour_cos",
] + [f"V{i}" for i in range(1, 29)]


class FeatureTransformer(BaseEstimator, TransformerMixin):
    """Transforms raw transaction records into model-ready features.

    Features generated:
    1. scaled_amount: RobustScaler on Amount (resistant to fraud spending outliers)
    2. log_amount: log1p(Amount) capturing non-linear monetary scaling
    3. hour_sin, hour_cos: Cyclical time representation of transaction hour
    4. V1 - V28: Raw PCA features
    """

    def __init__(self) -> None:
        self.amount_scaler = RobustScaler()
        self.is_fitted = False
        self.feature_columns = FEATURE_NAMES

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FeatureTransformer:
        """Fits scalers on training transactions."""
        df = X.copy()
        amounts = df[["Amount"]]
        self.amount_scaler.fit(amounts)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms input DataFrame into model features."""
        if not self.is_fitted:
            raise ValueError("FeatureTransformer must be fitted before transforming data.")

        df = X.copy()

        # 1. Cyclical time encoding (captures 24h diurnal pattern)
        if "Time" in df.columns:
            hours = (df["Time"] / 3600.0) % 24
            hour_sin = np.sin(2 * np.pi * hours / 24.0)
            hour_cos = np.cos(2 * np.pi * hours / 24.0)
        else:
            hour_sin = np.zeros(len(df))
            hour_cos = np.ones(len(df))

        # 2. Scaled and Log Amount
        scaled_amount = self.amount_scaler.transform(df[["Amount"]]).flatten()
        log_amount = np.log1p(np.maximum(0, df["Amount"].values))

        # 3. Assemble feature DataFrame
        features_dict = {
            "scaled_amount": scaled_amount,
            "log_amount": log_amount,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
        }

        for i in range(1, 29):
            col_name = f"V{i}"
            if col_name in df.columns:
                features_dict[col_name] = df[col_name].values
            else:
                features_dict[col_name] = np.zeros(len(df))

        feature_df = pd.DataFrame(features_dict, index=df.index)[self.feature_columns]
        return feature_df

    def transform_single(self, transaction: dict) -> pd.DataFrame:
        """Transforms a single transaction dict into a 1-row DataFrame."""
        df = pd.DataFrame([transaction])
        return self.transform(df)

    def save(self, filepath: Path | str | None = None) -> Path:
        """Saves the fitted transformer artifact."""
        if filepath is None:
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            filepath = ARTIFACTS_DIR / "preprocessor.joblib"
        else:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, filepath)
        logger.info("Saved FeatureTransformer to %s", filepath)
        return filepath

    @classmethod
    def load(cls, filepath: Path | str | None = None) -> FeatureTransformer:
        """Loads a pre-trained FeatureTransformer artifact."""
        if filepath is None:
            filepath = ARTIFACTS_DIR / "preprocessor.joblib"
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {filepath}")
        return joblib.load(filepath)
