"""
Offline retrospective changepoint detection using ruptures PELT.
Explicitly labeled offline/retrospective: MUST NOT be used for causal trading signals.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from regimeflow.logging_utils import logger


def run_pelt_changepoints(
    series: pd.Series | np.ndarray,
    model: str = "rbf",
    pen: float = 3.0,
    min_size: int = 21,
) -> Dict[str, Any]:
    """
    Run Pruned Exact Linear Time (PELT) changepoint segmentation.

    Parameters
    ----------
    series : pd.Series or np.ndarray
        Univariate or multivariate time series.
    model : str
        Cost model (e.g. "rbf", "l2", "normal").
    pen : float
        Penalty hyperparameter governing number of breakpoints.
    min_size : int
        Minimum segment length in sessions.

    Returns
    -------
    result : Dict[str, Any]
        Dictionary with detected breakpoints and retrospective offline warning.
    """
    try:
        import ruptures as rpt
    except ImportError:
        logger.warning("ruptures is not installed; skipping PELT changepoints.")
        return {
            "available": False,
            "error": "ruptures is not installed",
        }

    is_series = isinstance(series, pd.Series)
    index = series.index if is_series else None
    signal = np.asarray(series, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    algo = rpt.Pelt(model=model, min_size=min_size).fit(signal)
    breakpoint_indices = algo.predict(pen=pen)

    # Convert indices to timestamps if DatetimeIndex is available
    breakpoint_dates = []
    if index is not None:
        for idx in breakpoint_indices:
            # Note: ruptures end index is len(signal), so clip for date lookup
            capped_idx = min(idx - 1, len(index) - 1)
            breakpoint_dates.append(str(index[capped_idx].date()))

    logger.info(
        f"PELT (Offline) detected {len(breakpoint_indices) - 1} changepoints "
        f"(pen={pen}, model={model})."
    )

    return {
        "available": True,
        "is_causal_online": False,
        "semantics_warning": (
            "RETROSPECTIVE OFFLINE METHOD: PELT uses full-sample bilateral observations. "
            "It must NEVER generate real-time online trading signals."
        ),
        "model": model,
        "penalty": pen,
        "min_size": min_size,
        "n_changepoints": len(breakpoint_indices) - 1,
        "breakpoint_indices": breakpoint_indices,
        "breakpoint_dates": breakpoint_dates,
    }
