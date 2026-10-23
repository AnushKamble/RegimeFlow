"""
Deterministic synthetic data generators for HMM validation and financial time series.
"""

from typing import Tuple
import numpy as np
import pandas as pd


def generate_synthetic_hmm_data(
    n_samples: int,
    pi: np.ndarray,
    A: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic sequence from a known Gaussian Hidden Markov Model.

    Parameters
    ----------
    n_samples : int
        Number of time steps T.
    pi : np.ndarray of shape (K,)
        Initial state distribution (must sum to 1).
    A : np.ndarray of shape (K, K)
        Row-stochastic transition matrix.
    means : np.ndarray of shape (K, d)
        State emission means.
    covs : np.ndarray of shape (K, d, d)
        State emission covariance matrices (must be positive definite).
    seed : int
        Random seed for bitwise reproducibility.

    Returns
    -------
    X : np.ndarray of shape (n_samples, d)
        Emitted observations.
    states : np.ndarray of shape (n_samples,)
        Ground-truth latent state sequence in {0, ..., K-1}.
    """
    rng = np.random.default_rng(seed)
    n_states = len(pi)
    dim = means.shape[1]

    states = np.zeros(n_samples, dtype=int)
    X = np.zeros((n_samples, dim), dtype=np.float64)

    # Initial state
    states[0] = rng.choice(n_states, p=pi)
    X[0] = rng.multivariate_normal(means[states[0]], covs[states[0]])

    # Sequence evolution
    for t in range(1, n_samples):
        prev_s = states[t - 1]
        states[t] = rng.choice(n_states, p=A[prev_s])
        X[t] = rng.multivariate_normal(means[states[t]], covs[states[t]])

    return X, states


def generate_synthetic_market_data(
    n_sessions: int = 1500,
    start_date: str = "2010-01-04",
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate realistic synthetic daily OHLCV market data driven by a 3-regime process:
    - State 0: Low-volatility Bull market (positive drift, low daily vol ~10% annualized)
    - State 1: Medium-volatility Sideways/Choppy market (near-zero drift, medium vol ~18% annualized)
    - State 2: High-volatility Crash/Panic market (negative drift, high vol ~40% annualized)

    Parameters
    ----------
    n_sessions : int
        Number of trading sessions.
    start_date : str
        Start date ISO string.
    seed : int
        Random seed.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with datetime index and columns ['open', 'high', 'low', 'close', 'adj_close', 'volume'].
    states : np.ndarray
        Underlying true regime sequence of length n_sessions.
    """
    rng = np.random.default_rng(seed)

    # Transition matrix with realistic persistence
    A = np.array([
        [0.97, 0.025, 0.005],  # State 0 (Bull): highly persistent
        [0.05, 0.90, 0.05],    # State 1 (Sideways): moderately persistent
        [0.03, 0.07, 0.90],    # State 2 (Crash): persistent crisis clustering
    ], dtype=np.float64)

    # State parameters: daily drift and daily vol
    # Annualized: Bull (12% ret, 10% vol), Sideways (2% ret, 18% vol), Crash (-35% ret, 45% vol)
    daily_mu = np.array([0.12 / 252.0, 0.02 / 252.0, -0.35 / 252.0])
    daily_sigma = np.array([0.10 / np.sqrt(252), 0.18 / np.sqrt(252), 0.45 / np.sqrt(252)])

    states = np.zeros(n_sessions, dtype=int)
    states[0] = 0

    for t in range(1, n_sessions):
        states[t] = rng.choice(3, p=A[states[t - 1]])

    # Generate daily log returns
    eps = rng.standard_normal(n_sessions)
    # Heavy tails via t-distribution mix in crash regime
    t_innovations = rng.standard_t(df=4, size=n_sessions) * np.sqrt(2 / 4)
    regime_t_weight = np.where(states == 2, 0.4, 0.0)
    composite_innovations = (1.0 - regime_t_weight) * eps + regime_t_weight * t_innovations

    returns = daily_mu[states] + daily_sigma[states] * composite_innovations

    # Build prices starting at $100
    prices = np.zeros(n_sessions, dtype=np.float64)
    prices[0] = 100.0
    for t in range(1, n_sessions):
        prices[t] = prices[t - 1] * np.exp(returns[t])

    # Generate realistic OHLC bars around close price
    # Intraday range scales with daily volatility
    intraday_vol = daily_sigma[states]
    open_prices = np.zeros(n_sessions, dtype=np.float64)
    high_prices = np.zeros(n_sessions, dtype=np.float64)
    low_prices = np.zeros(n_sessions, dtype=np.float64)
    close_prices = prices.copy()

    # Day 0 open
    open_prices[0] = 100.0 * (1.0 + rng.uniform(-0.002, 0.002))

    for t in range(n_sessions):
        if t > 0:
            # Open is close[t-1] plus small overnight jump
            overnight_gap = rng.normal(0, intraday_vol[t] * 0.3)
            open_prices[t] = close_prices[t - 1] * np.exp(overnight_gap)

        bar_max = max(open_prices[t], close_prices[t])
        bar_min = min(open_prices[t], close_prices[t])

        # High is above bar_max, Low is below bar_min
        high_bump = np.abs(rng.normal(0, intraday_vol[t] * 0.8))
        low_dip = np.abs(rng.normal(0, intraday_vol[t] * 0.8))

        high_prices[t] = bar_max * (1.0 + high_bump)
        low_prices[t] = bar_min * max(0.001, (1.0 - low_dip))
        # Ensure strict inequality
        high_prices[t] = max(high_prices[t], bar_max * 1.0001)
        low_prices[t] = min(low_prices[t], bar_min * 0.9999)

    # Volume: elevated volume in high-volatility regimes
    base_volume = np.array([50_000_000, 70_000_000, 130_000_000])
    volume_noise = np.exp(rng.normal(0, 0.25, size=n_sessions))
    volume = (base_volume[states] * volume_noise).astype(np.float64)

    # Business dates index
    dates = pd.bdate_range(start=start_date, periods=n_sessions)

    # Adjusted close identical to close for synthetic series (no split/dividend)
    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "adj_close": close_prices,
            "volume": volume,
        },
        index=dates,
    )
    df.index.name = "date"
    return df, states
