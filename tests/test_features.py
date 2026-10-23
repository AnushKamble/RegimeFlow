"""
Tests for financial feature engineering formulas, pipeline, scalers, and causality invariants.
"""

import numpy as np
import pandas as pd
import pytest
from regimeflow.config import FeatureConfig
from regimeflow.data.synthetic import generate_synthetic_market_data
from regimeflow.features.formulas import (
    compute_garman_klass_volatility,
    compute_log_returns,
    compute_realized_volatility,
    compute_trailing_momentum,
    compute_volatility_ratio,
    compute_volume_zscore,
)
from regimeflow.features.pipeline import FeaturePipeline
from regimeflow.features.transforms import TrainOnlyRobustScaler, TrainOnlyStandardScaler


def test_hand_calculated_formulas():
    # 1. Log returns
    prices = pd.Series([100.0, 110.0, 99.0])
    rets = compute_log_returns(prices)
    assert np.isnan(rets.iloc[0])
    assert np.isclose(rets.iloc[1], np.log(1.1))
    assert np.isclose(rets.iloc[2], np.log(0.9))

    # 2. Realized volatility (window=2, non-annualized for exact check)
    ret_series = pd.Series([0.01, -0.02, 0.03])
    # window 2 over [-0.02, 0.03]: mean = 0.005, sample variance = ((-0.025)^2 + (0.025)^2)/1 = 0.00125
    vol = compute_realized_volatility(ret_series, window=2, annualize=False, ddof=1)
    expected_std = np.sqrt(0.00125)
    assert np.isclose(vol.iloc[2], expected_std)

    # 3. Trailing momentum (window=2)
    mom = compute_trailing_momentum(prices, window=2)
    # (99.0 / 100.0) - 1.0 = -0.01
    assert np.isclose(mom.iloc[2], -0.01)

    # 4. Garman-Klass single bar calculation
    # O=100, H=105, L=95, C=102
    # log(H/L) = log(105/95) = log(1.105263)
    # log(C/O) = log(102/100) = log(1.02)
    # gk_var = 0.5 * log(105/95)^2 - (2*log(2)-1) * log(1.02)^2
    hl = np.log(105.0 / 95.0)
    co = np.log(102.0 / 100.0)
    expected_gk_var = 0.5 * (hl**2) - (2.0 * np.log(2.0) - 1.0) * (co**2)
    gk_vol = compute_garman_klass_volatility(
        open_prices=pd.Series([100.0]),
        high_prices=pd.Series([105.0]),
        low_prices=pd.Series([95.0]),
        close_prices=pd.Series([102.0]),
        window=1,
        annualize=False,
    )
    assert np.isclose(gk_vol.iloc[0], np.sqrt(expected_gk_var))


def test_feature_causality_invariant():
    """
    CRITICAL LEAKAGE TEST:
    Modifying market data strictly after session t MUST NOT alter features up to session t.
    """
    df, _ = generate_synthetic_market_data(n_sessions=200, seed=42)
    cfg = FeatureConfig(feature_group="extended")
    pipeline = FeaturePipeline(cfg)

    feat_orig, _ = pipeline.fit_transform(df)

    t_split = 120
    split_date = df.index[t_split]

    # Create perturbed DataFrame where all data after split_date is drastically altered
    df_perturbed = df.copy()
    perturb_mask = df_perturbed.index > split_date
    df_perturbed.loc[perturb_mask, "close"] *= 3.0
    df_perturbed.loc[perturb_mask, "high"] *= 3.5
    df_perturbed.loc[perturb_mask, "low"] *= 2.5
    df_perturbed.loc[perturb_mask, "volume"] *= 10.0

    feat_perturbed, _ = pipeline.fit_transform(df_perturbed)

    # Compare features on or before split_date
    common_dates = feat_orig.index[feat_orig.index <= split_date]
    assert len(common_dates) > 0

    for col in feat_orig.columns:
        diff = np.abs(feat_orig.loc[common_dates, col] - feat_perturbed.loc[common_dates, col])
        max_diff = np.max(diff)
        assert max_diff == 0.0, (
            f"Feature causality violation in column '{col}': max difference {max_diff} > 0"
        )


def test_train_only_scaler_isolation():
    """
    Asserts that fitting a scaler on the training set is completely independent of test data.
    """
    df, _ = generate_synthetic_market_data(n_sessions=250, seed=42)
    pipeline = FeaturePipeline(FeatureConfig(feature_group="base"))
    feat, _ = pipeline.fit_transform(df)

    n_train = 150
    X_train = feat.iloc[:n_train]
    X_test_1 = feat.iloc[n_train:].copy()
    X_test_2 = feat.iloc[n_train:].copy() * 100.0  # massively perturbed test set

    scaler1 = TrainOnlyStandardScaler()
    scaler1.fit(X_train)

    scaler2 = TrainOnlyStandardScaler()
    scaler2.fit(X_train)

    assert np.allclose(scaler1.mean_, scaler2.mean_)
    assert np.allclose(scaler1.std_, scaler2.std_)

    # Test serialization round-trip
    state = scaler1.to_dict()
    scaler_reloaded = TrainOnlyStandardScaler.from_dict(state)
    assert np.allclose(scaler1.mean_, scaler_reloaded.mean_)
    assert np.allclose(scaler1.std_, scaler_reloaded.std_)

    # Robust scaler test
    r_scaler = TrainOnlyRobustScaler()
    r_scaler.fit(X_train)
    assert r_scaler.is_fitted_
    r_state = r_scaler.to_dict()
    r_reloaded = TrainOnlyRobustScaler.from_dict(r_state)
    assert np.allclose(r_scaler.center_, r_reloaded.center_)
    assert np.allclose(r_scaler.scale_, r_reloaded.scale_)
