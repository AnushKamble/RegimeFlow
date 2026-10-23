"""
Mathematical formulas for financial feature engineering.
All formulas strictly evaluate backward-looking trailing windows.
"""

import numpy as np
import pandas as pd


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Daily log returns: r_t = log(P_t / P_{t-1})
    """
    return np.log(prices / prices.shift(1))


def compute_realized_volatility(
    returns: pd.Series,
    window: int = 21,
    annualize: bool = True,
    ddof: int = 1,
) -> pd.Series:
    """
    Rolling realized volatility over trailing `window` sessions.
    """
    roll_std = returns.rolling(window=window, min_periods=window).std(ddof=ddof)
    if annualize:
        return roll_std * np.sqrt(252.0)
    return roll_std


def compute_volatility_ratio(
    short_vol: pd.Series,
    long_vol: pd.Series,
    eps: float = 1e-8,
) -> pd.Series:
    """
    Short-term to long-term volatility ratio: RV_short / max(RV_long, eps).
    """
    denom = np.maximum(long_vol, eps)
    return short_vol / denom


def compute_trailing_momentum(prices: pd.Series, window: int = 21) -> pd.Series:
    """
    Trailing percentage return momentum over `window` sessions: P_t / P_{t-window} - 1.
    """
    return (prices / prices.shift(window)) - 1.0


def compute_garman_klass_volatility(
    open_prices: pd.Series,
    high_prices: pd.Series,
    low_prices: pd.Series,
    close_prices: pd.Series,
    window: int = 5,
    annualize: bool = True,
    eps: float = 1e-8,
) -> pd.Series:
    """
    Garman-Klass range-based volatility:
    GKVar_t = 0.5 * [log(H_t/L_t)]^2 - (2*log(2)-1) * [log(C_t/O_t)]^2
    Rolling average over `window` sessions, annualized as sqrt(mean * 252).
    Negative single-bar variances (caused by finite discretizations) are floored at eps.
    """
    log_hl = np.log(high_prices / low_prices)
    log_co = np.log(close_prices / open_prices)
    gk_var_raw = 0.5 * (log_hl**2) - (2.0 * np.log(2.0) - 1.0) * (log_co**2)

    # Floor individual bar variance
    gk_var = np.maximum(gk_var_raw, eps)

    # Rolling window mean
    gk_roll = gk_var.rolling(window=window, min_periods=window).mean()
    gk_roll = np.maximum(gk_roll, eps)

    if annualize:
        return np.sqrt(gk_roll * 252.0)
    return np.sqrt(gk_roll)


def compute_parkinson_volatility(
    high_prices: pd.Series,
    low_prices: pd.Series,
    window: int = 5,
    annualize: bool = True,
    eps: float = 1e-8,
) -> pd.Series:
    """
    Parkinson daily range variance: [log(H/L)]^2 / (4 * log(2)).
    """
    log_hl = np.log(high_prices / low_prices)
    park_var = (log_hl**2) / (4.0 * np.log(2.0))
    park_roll = park_var.rolling(window=window, min_periods=window).mean()
    park_roll = np.maximum(park_roll, eps)

    if annualize:
        return np.sqrt(park_roll * 252.0)
    return np.sqrt(park_roll)


def compute_volume_zscore(
    volume: pd.Series,
    window: int = 63,
    eps: float = 1e-8,
) -> pd.Series:
    """
    Trailing volume z-score using log volume:
    LV_t = log(1 + Volume_t)
    ZVol_t = (LV_t - mean_{63, t}) / max(std_{63, t}, eps)
    """
    log_vol = np.log1p(np.maximum(volume, 0.0))
    roll_mean = log_vol.rolling(window=window, min_periods=window).mean()
    roll_std = log_vol.rolling(window=window, min_periods=window).std(ddof=1)
    denom = np.maximum(roll_std, eps)
    return (log_vol - roll_mean) / denom
