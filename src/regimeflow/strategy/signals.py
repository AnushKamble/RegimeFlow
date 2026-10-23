"""
Signal generation for benchmark strategies and regime-aware allocation.
Strictly generates target weights w_t using information available at or before close t.
"""

import numpy as np
import pandas as pd


def generate_buy_and_hold_signals(index: pd.DatetimeIndex) -> pd.Series:
    """Buy & Hold benchmark: target weight = 1.0 for all t."""
    return pd.Series(1.0, index=index, name="weight_buy_and_hold")


def generate_moving_average_signals(
    prices: pd.Series,
    window: int = 200,
) -> pd.Series:
    """
    200-day Simple Moving Average trend baseline:
    w_t = 1.0 if P_t >= SMA_200(P_t) else 0.0.
    """
    sma = prices.rolling(window=window, min_periods=window).mean()
    weights = np.where(prices >= sma, 1.0, 0.0)
    # Fill warmup with 0.0 (or cash)
    weights[np.isnan(sma)] = 0.0
    return pd.Series(weights, index=prices.index, name="weight_sma_200")


def generate_regime_signals(
    high_risk_prob: pd.Series,
    multiplier: float = 1.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> pd.Series:
    """
    Regime-aware risk control exposure policy:
    w_t = clip(1.0 - lambda * q_t, min_weight, max_weight)
    where q_t is the causal filtered high-risk state probability at close t.
    """
    raw_weights = 1.0 - multiplier * high_risk_prob
    clipped_weights = np.clip(raw_weights, min_weight, max_weight)
    return pd.Series(clipped_weights, index=high_risk_prob.index, name="weight_regime")
