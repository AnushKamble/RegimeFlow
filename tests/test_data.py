"""
Unit and integration tests for data generation, validation, caching, and providers.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from regimeflow.config import DataConfig
from regimeflow.data.cache import CacheManager
from regimeflow.data.providers import DataPipeline, SyntheticProvider
from regimeflow.data.schema import DatasetManifest
from regimeflow.data.synthetic import (
    generate_synthetic_hmm_data,
    generate_synthetic_market_data,
)
from regimeflow.data.validation import validate_market_data


def test_synthetic_hmm_generator():
    pi = np.array([0.6, 0.4])
    A = np.array([[0.9, 0.1], [0.2, 0.8]])
    means = np.array([[0.01, 0.02], [-0.02, 0.05]])
    covs = np.array([
        [[0.001, 0.0], [0.0, 0.002]],
        [[0.003, 0.0], [0.0, 0.004]],
    ])

    X1, s1 = generate_synthetic_hmm_data(100, pi, A, means, covs, seed=42)
    X2, s2 = generate_synthetic_hmm_data(100, pi, A, means, covs, seed=42)

    assert X1.shape == (100, 2)
    assert s1.shape == (100,)
    assert np.allclose(X1, X2), "Synthetic HMM must be deterministic with identical seed"
    assert np.array_equal(s1, s2)
    assert set(s1).issubset({0, 1})


def test_synthetic_market_generator_and_validation():
    df, states = generate_synthetic_market_data(n_sessions=500, seed=123)
    assert len(df) == 500
    assert len(states) == 500
    assert list(df.columns) == ["open", "high", "low", "close", "adj_close", "volume"]

    report = validate_market_data(df, min_sessions=100)
    assert report.is_valid is True
    assert len(report.errors) == 0
    assert report.row_count == 500


def test_data_validation_failures():
    df, _ = generate_synthetic_market_data(n_sessions=200, seed=42)

    # 1. Non-positive price
    df_bad = df.copy()
    df_bad.iloc[10, df_bad.columns.get_loc("close")] = -5.0
    report = validate_market_data(df_bad)
    assert report.is_valid is False
    assert any("non-positive" in err for err in report.errors)

    # 2. Bar consistency (low > open and close)
    df_bad = df.copy()
    df_bad.iloc[5, df_bad.columns.get_loc("low")] = df_bad.iloc[5]["high"] + 10.0
    report = validate_market_data(df_bad)
    assert report.is_valid is False
    assert any("Low > min(Open, Close)" in err for err in report.errors)

    # 3. Duplicate dates
    df_bad = df.copy()
    idx = list(df_bad.index)
    idx[1] = idx[0]
    df_bad.index = idx
    report = validate_market_data(df_bad)
    assert report.is_valid is False
    assert any("duplicate" in err.lower() for err in report.errors)

    # 4. NaN values
    df_bad = df.copy()
    df_bad.iloc[3, df_bad.columns.get_loc("volume")] = np.nan
    report = validate_market_data(df_bad)
    assert report.is_valid is False
    assert any("non-finite" in err for err in report.errors)


def test_cache_manager_sha256_and_corruption_detection(tmp_path):
    cache = CacheManager(cache_dir=tmp_path)
    df, _ = generate_synthetic_market_data(n_sessions=150, seed=99)
    manifest = DatasetManifest(
        symbol="TEST",
        source="synthetic",
        row_count=len(df),
    )

    dataset_id = "test_dataset"
    cache.save(dataset_id, df, manifest)

    # Verify loading works
    loaded_df, loaded_manifest = cache.load(dataset_id)
    assert loaded_df is not None
    assert loaded_manifest is not None
    assert len(loaded_df) == 150
    assert loaded_manifest.symbol == "TEST"

    # Simulate corruption by altering the parquet file content
    data_path, _ = cache.get_paths(dataset_id)
    with open(data_path, "ab") as f:
        f.write(b"corrupting_bytes")

    corrupted_df, corrupted_manifest = cache.load(dataset_id)
    assert corrupted_df is None, "Corrupted cache must return None"
    assert corrupted_manifest is None


def test_data_pipeline_synthetic_flow(tmp_path):
    cache = CacheManager(cache_dir=tmp_path)
    pipeline = DataPipeline(cache_manager=cache)

    config = DataConfig(
        symbol="SYNTH",
        source="synthetic",
        synthetic_length=300,
        cache_enabled=True,
        seed=77,
    )

    # First call: generates, validates, caches
    df1, manifest1, report1 = pipeline.load_or_fetch(config)
    assert report1.is_valid is True
    assert len(df1) == 300

    # Second call: loads from cache
    df2, manifest2, report2 = pipeline.load_or_fetch(config)
    assert len(df2) == 300
    assert np.allclose(df1["close"], df2["close"])
    assert manifest1.sha256_hash == manifest2.sha256_hash
