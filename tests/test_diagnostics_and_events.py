"""
Tests for state profiles, entropy, dwell times, and event window diagnostics.
"""

import numpy as np
import pandas as pd
import pytest
from regimeflow.config import EventWindowConfig
from regimeflow.metrics.events import analyze_event_windows
from regimeflow.models.gaussian_hmm import GaussianHMM
from regimeflow.states.profiles import compute_posterior_entropy, compute_state_profiles


def test_state_profiles_and_dwell_times():
    X = np.random.default_rng(42).normal(size=(200, 2))
    model = GaussianHMM(n_states=2, covariance_type="full", n_restarts=1, max_iter=15)
    model.fit(X)

    gamma = model.smooth(X)
    profiles = compute_state_profiles(model, X, gamma, feature_names=["ret", "vol"])

    assert len(profiles) == 2
    assert "occupancy_fraction" in profiles.columns
    assert "persistence_A_kk" in profiles.columns
    assert "expected_dwell_sessions" in profiles.columns
    assert "ret_mean" in profiles.columns
    assert "vol_std" in profiles.columns

    # Occupancies sum to 1
    assert np.isclose(profiles["occupancy_fraction"].sum(), 1.0)
    # Expected dwell time is >= 1.0
    assert (profiles["expected_dwell_sessions"] >= 1.0).all()


def test_posterior_entropy():
    # Certain state: p = [1.0, 0.0] -> entropy = 0.0
    certain_p = np.array([[1.0, 0.0], [0.0, 1.0]])
    ent_certain = compute_posterior_entropy(certain_p)
    assert np.allclose(ent_certain, 0.0)

    # Maximum uncertainty: p = [0.5, 0.5] -> normalized entropy = 1.0
    uncertain_p = np.array([[0.5, 0.5]])
    ent_uncertain = compute_posterior_entropy(uncertain_p)
    assert np.isclose(ent_uncertain[0], 1.0)


def test_event_window_analysis():
    dates = pd.bdate_range("2020-01-01", periods=100)
    # Simulate high risk probability spiking around session 40-60 (e.g. Feb-March 2020)
    risk_probs = np.full(100, 0.1)
    risk_probs[35:60] = 0.85
    fwd_rets = np.full(100, 0.0005)
    fwd_rets[35:60] = -0.015

    oos_df = pd.DataFrame(
        {
            "high_risk_prob": risk_probs,
            "forward_asset_return": fwd_rets,
        },
        index=dates,
    )

    event_start = str(dates[35].date())
    event_end = str(dates[55].date())
    evt_cfg = [
        EventWindowConfig(
            name="TEST_SHOCK",
            start_date=event_start,
            end_date=event_end,
            description="Synthetic Market Shock",
        )
    ]

    event_summary = analyze_event_windows(oos_df, evt_cfg, risk_threshold=0.5)
    assert len(event_summary) == 1
    row = event_summary.iloc[0]
    assert row["event_name"] == "TEST_SHOCK"
    assert row["mean_risk_prob"] > 0.7
    assert row["pre_event_baseline_risk_prob"] < 0.2
    assert row["elevated_risk_fraction"] > 0.9
    assert row["asset_max_drawdown"] < 0.0
