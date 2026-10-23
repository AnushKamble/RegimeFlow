"""
Backtest engine with explicit execution lag and turnover-based transaction cost accounting.
"""

from typing import Tuple
import numpy as np
import pandas as pd


class BacktestEngine:
    """
    Simulates portfolio strategy with strict one-period signal-to-execution lag
    and proportional turnover transaction costs.
    """

    def __init__(
        self,
        transaction_cost_bps: float = 10.0,
        cash_return_annual: float = 0.0,
    ):
        self.cost_rate = transaction_cost_bps / 10_000.0  # 10 bps = 0.0010
        self.daily_cash_return = cash_return_annual / 252.0

    def run(
        self,
        weights: pd.Series,
        forward_asset_returns: pd.Series,
    ) -> pd.DataFrame:
        """
        Execute backtest.

        Parameters
        ----------
        weights : pd.Series
            Target portfolio exposure w_t established at close t.
        forward_asset_returns : pd.Series
            Asset return earned over the subsequent session: r_{t+1} = (P_{t+1} / P_t) - 1.

        Returns
        -------
        results_df : pd.DataFrame
            DataFrame with columns:
            ['weight', 'turnover', 'transaction_cost', 'gross_return', 'net_return', 'nav_gross', 'nav_net'].
        """
        # Align index
        common_idx = weights.index.intersection(forward_asset_returns.index)
        w = weights.loc[common_idx].to_numpy(dtype=np.float64)
        r_asset = forward_asset_returns.loc[common_idx].to_numpy(dtype=np.float64)

        # Drop any trailing NaN where forward return is not yet realized
        valid_mask = ~np.isnan(r_asset)
        w = w[valid_mask]
        r_asset = r_asset[valid_mask]
        valid_idx = common_idx[valid_mask]

        T = len(w)
        if T == 0:
            return pd.DataFrame()

        # Compute turnover: Delta w_t = |w_t - w_{t-1}| with w_{-1} = 0.0 (initial cash entry)
        prev_w = np.zeros(T, dtype=np.float64)
        prev_w[1:] = w[:-1]
        turnover = np.abs(w - prev_w)

        # Transaction costs paid at entry/rebalance
        costs = self.cost_rate * turnover

        # Gross & Net returns
        # w_t earns r_asset_{t+1}
        gross_returns = w * r_asset + (1.0 - w) * self.daily_cash_return
        net_returns = gross_returns - costs

        # Cumulative NAV
        nav_gross = np.cumprod(1.0 + gross_returns)
        nav_net = np.cumprod(1.0 + net_returns)

        return pd.DataFrame(
            {
                "weight": w,
                "turnover": turnover,
                "transaction_cost": costs,
                "gross_return": gross_returns,
                "net_return": net_returns,
                "nav_gross": nav_gross,
                "nav_net": nav_net,
            },
            index=valid_idx,
        )
