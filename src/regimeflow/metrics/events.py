"""
Predeclared historical event-window analysis for market stress regimes.
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd
from regimeflow.config import EventWindowConfig
from regimeflow.logging_utils import logger


def analyze_event_windows(
    oos_predictions: pd.DataFrame,
    event_configs: List[EventWindowConfig],
    risk_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Evaluate causal filtered risk probabilities and drawdown dynamics
    during locked historical stress windows.

    Parameters
    ----------
    oos_predictions : pd.DataFrame
        Timestamped OOS predictions with columns ['high_risk_prob', 'forward_asset_return'].
    event_configs : List[EventWindowConfig]
        Predeclared event windows.
    risk_threshold : float
        Probability threshold above which state is flagged as high risk.

    Returns
    -------
    event_metrics_df : pd.DataFrame
        Summary table for each event window.
    """
    records = []

    for evt in event_configs:
        start_dt = pd.to_datetime(evt.start_date)
        end_dt = pd.to_datetime(evt.end_date)

        # Slice event window
        event_mask = (oos_predictions.index >= start_dt) & (oos_predictions.index <= end_dt)
        event_df = oos_predictions.loc[event_mask]

        if event_df.empty:
            logger.info(f"Event window '{evt.name}' [{evt.start_date} -> {evt.end_date}] not in OOS range.")
            continue

        n_sessions = len(event_df)
        mean_risk_prob = float(event_df["high_risk_prob"].mean())
        peak_risk_prob = float(event_df["high_risk_prob"].max())
        elevated_fraction = float(np.mean(event_df["high_risk_prob"] >= risk_threshold))

        # Lookback baseline (63 sessions before event start)
        pre_event_mask = (oos_predictions.index < start_dt) & (
            oos_predictions.index >= start_dt - pd.Timedelta(days=120)
        )
        pre_event_df = oos_predictions.loc[pre_event_mask]
        baseline_risk_prob = (
            float(pre_event_df["high_risk_prob"].mean()) if not pre_event_df.empty else float("nan")
        )

        # Asset returns and drawdown
        returns = event_df["forward_asset_return"].dropna()
        if not returns.empty:
            cum_returns = np.exp(np.cumsum(returns.to_numpy()))
            running_max = np.maximum.accumulate(cum_returns)
            drawdowns = (cum_returns - running_max) / running_max
            max_drawdown = float(np.min(drawdowns))
            total_return = float(cum_returns[-1] - 1.0)
        else:
            max_drawdown = float("nan")
            total_return = float("nan")

        rec: Dict[str, Any] = {
            "event_name": evt.name,
            "description": evt.description,
            "start_date": evt.start_date,
            "end_date": evt.end_date,
            "n_sessions": n_sessions,
            "mean_risk_prob": mean_risk_prob,
            "pre_event_baseline_risk_prob": baseline_risk_prob,
            "peak_risk_prob": peak_risk_prob,
            "elevated_risk_fraction": elevated_fraction,
            "asset_total_return": total_return,
            "asset_max_drawdown": max_drawdown,
        }
        records.append(rec)

    if not records:
        return pd.DataFrame(
            columns=[
                "event_name",
                "description",
                "start_date",
                "end_date",
                "n_sessions",
                "mean_risk_prob",
                "pre_event_baseline_risk_prob",
                "peak_risk_prob",
                "elevated_risk_fraction",
                "asset_total_return",
                "asset_max_drawdown",
            ]
        )

    return pd.DataFrame.from_records(records)
