"""
Financial performance and risk metrics computation.
"""

from typing import Dict
import numpy as np
import pandas as pd


def compute_financial_metrics(
    backtest_result: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """
    Calculate annualized return, risk, drawdown, turnover, and cost metrics.

    Parameters
    ----------
    backtest_result : pd.DataFrame
        Output from BacktestEngine with columns:
        ['net_return', 'gross_return', 'turnover', 'transaction_cost', 'weight', 'nav_net'].
    risk_free_rate : float
        Annualized risk-free rate (default 0.0).

    Returns
    -------
    metrics : Dict[str, float]
    """
    if backtest_result.empty:
        return {}

    N = len(backtest_result)
    net_rets = backtest_result["net_return"].to_numpy()
    gross_rets = backtest_result["gross_return"].to_numpy()
    nav_net = backtest_result["nav_net"].to_numpy()
    weights = backtest_result["weight"].to_numpy()
    turnovers = backtest_result["turnover"].to_numpy()
    costs = backtest_result["transaction_cost"].to_numpy()

    # Total Return
    total_net_return = float(nav_net[-1] - 1.0)
    total_gross_return = float(np.prod(1.0 + gross_rets) - 1.0)

    # CAGR: (NAV_T / NAV_0)^(252 / N) - 1
    # Guard against negative NAV
    if nav_net[-1] > 0:
        cagr = float((nav_net[-1]) ** (252.0 / N) - 1.0)
    else:
        cagr = -1.0

    # Annualized Volatility
    daily_vol = np.std(net_rets, ddof=1) if N > 1 else 0.0
    ann_vol = float(daily_vol * np.sqrt(252.0))

    # Sharpe Ratio: (CAGR - rf) / AnnVol
    sharpe = float((cagr - risk_free_rate) / ann_vol) if ann_vol > 1e-8 else 0.0

    # Maximum Drawdown
    running_max = np.maximum.accumulate(nav_net)
    drawdowns = (nav_net - running_max) / running_max
    max_drawdown = float(np.min(drawdowns))

    # Calmar Ratio
    calmar = float(cagr / abs(max_drawdown)) if abs(max_drawdown) > 1e-8 else 0.0

    # Exposure & Turnover
    mean_exposure = float(np.mean(weights))
    total_turnover = float(np.sum(turnovers))
    annual_turnover = float(total_turnover * (252.0 / N))
    total_cost_pct = float(np.sum(costs))

    # Hit Rate (% positive sessions)
    hit_rate = float(np.mean(net_rets > 0))

    return {
        "n_sessions": N,
        "total_net_return": total_net_return,
        "total_gross_return": total_gross_return,
        "cagr": cagr,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "mean_exposure": mean_exposure,
        "total_turnover": total_turnover,
        "annualized_turnover": annual_turnover,
        "total_cost_paid": total_cost_pct,
        "hit_rate": hit_rate,
    }
