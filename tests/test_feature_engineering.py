"""Unit tests for FeatureTransformer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from training.feature_engineering import FeatureTransformer


def _make_dummy_df(n: int = 100) -> pd.DataFrame:
    data = {
        "Time": np.linspace(0, 172800, n),
        "Amount": np.random.exponential(scale=50.0, size=n),
        "Class": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
    }
    for i in range(1, 29):
        data[f"V{i}"] = np.random.randn(n)
    return pd.DataFrame(data)


def test_feature_transformer_fit_transform():
    df = _make_dummy_df(100)
    transformer = FeatureTransformer()
    transformer.fit(df)

    transformed = transformer.transform(df)

    assert "scaled_amount" in transformed.columns
    assert "log_amount" in transformed.columns
    assert "hour_sin" in transformed.columns
    assert "hour_cos" in transformed.columns
    assert "Class" not in transformed.columns
    assert transformed.shape[0] == 100
    assert not transformed.isna().any().any()


def test_feature_transformer_single_transform():
    df = _make_dummy_df(50)
    transformer = FeatureTransformer().fit(df)

    single_row = {
        "Time": 3600.0,
        "Amount": 100.0,
    }
    for i in range(1, 29):
        single_row[f"V{i}"] = 0.5

    out_df = transformer.transform_single(single_row)
    assert out_df.shape[0] == 1
    assert set(out_df.columns) == set(transformer.feature_columns)


def test_feature_transformer_persistence(tmp_path):
    df = _make_dummy_df(50)
    transformer = FeatureTransformer().fit(df)

    save_file = tmp_path / "preprocessor.joblib"
    transformer.save(save_file)

    loaded = FeatureTransformer.load(save_file)
    assert loaded.feature_columns == transformer.feature_columns
