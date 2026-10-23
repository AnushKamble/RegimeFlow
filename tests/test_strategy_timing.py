"""
Tests for backtest execution timing, signal lags, and transaction cost accounting.
"""

import numpy as np
import pandas as pd
import pytest
from regimeflow.metrics.financial import compute_financial_metrics
from regimeflow.strategy.backtest import BacktestEngine
from regimeflow.strategy.signals import (
    generate_buy_and_hold_signals,
    generate_moving_average_signals,
    generate_regime_signals,
)


def test_hand_calculated_5_row_backtest():
    """
    CRITICAL AUDIT TEST:
    Verify exact hand-calculated numbers for weights, turnover, costs, and net returns.
    """
    dates = pd.bdate_range("2020-01-01", periods=3)
    # Day 0: weight 1.0, forward return 0.02
    # Day 1: weight 0.5, forward return -0.01
    # Day 2: weight 0.0, forward return -0.05
    weights = pd.Series([1.0, 0.5, 0.0], index=dates)
    fwd_returns = pd.Series([0.02, -0.01, -0.05], index=dates)

    # Cost = 10 bps = 0.0010
    engine = BacktestEngine(transaction_cost_bps=10.0, cash_return_annual=0.0)
    res = engine.run(weights, fwd_returns)

    # Day 0:
    # Initial turnover from cash w_{-1}=0 to w_0=1.0: turnover = 1.0, cost = 0.0010
    # Gross = 1.0 * 0.02 = 0.02. Net = 0.02 - 0.001 = 0.019
    assert np.isclose(res.iloc[0]["turnover"], 1.0)
    assert np.isclose(res.iloc[0]["transaction_cost"], 0.0010)
    assert np.isclose(res.iloc[0]["gross_return"], 0.02)
    assert np.isclose(res.iloc[0]["net_return"], 0.0190)

    # Day 1:
    # Turnover = |0.5 - 1.0| = 0.5. Cost = 0.5 * 0.0010 = 0.0005
    # Gross = 0.5 * (-0.01) = -0.005. Net = -0.005 - 0.0005 = -0.0055
    assert np.isclose(res.iloc[1]["turnover"], 0.5)
    assert np.isclose(res.iloc[1]["transaction_cost"], 0.0005)
    assert np.isclose(res.iloc[1]["gross_return"], -0.0050)
    assert np.isclose(res.iloc[1]["net_return"], -0.0055)

    # Day 2:
    # Turnover = |0.0 - 0.5| = 0.5. Cost = 0.0005
    # Gross = 0.0. Net = 0.0 - 0.0005 = -0.0005
    assert np.isclose(res.iloc[2]["turnover"], 0.5)
    assert np.isclose(res.iloc[2]["transaction_cost"], 0.0005)
    assert np.isclose(res.iloc[2]["gross_return"], 0.0)
    assert np.isclose(res.iloc[2]["net_return"], -0.0005)

    # Metrics
    metrics = compute_financial_metrics(res)
    assert metrics["n_sessions"] == 3
    assert np.isclose(metrics["total_turnover"], 2.0)
    assert np.isclose(metrics["total_cost_paid"], 0.0020)


def test_zero_exposure_earns_cash():
    """Verify that 0% exposure earns cash rate, never underlying asset moves."""
    dates = pd.bdate_range("2021-01-01", periods=5)
    weights = pd.Series([0.0] * 5, index=dates)
    # Huge asset crashes
    fwd_returns = pd.Series([-0.10, -0.20, -0.15, -0.05, -0.08], index=dates)

    engine = BacktestEngine(transaction_cost_bps=0.0, cash_return_annual=0.0)
    res = engine.run(weights, fwd_returns)

    assert (res["gross_return"] == 0.0).all()
    assert (res["net_return"] == 0.0).all()
    assert np.isclose(res["nav_net"].iloc[-1], 1.0)
