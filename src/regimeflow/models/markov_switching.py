"""
Markov-Switching Regression comparator using statsmodels.
Compares filtered vs smoothed probabilities on financial returns.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from regimeflow.logging_utils import logger


def fit_markov_switching_regression(
    returns: pd.Series | np.ndarray,
    k_regimes: int = 2,
    switching_variance: bool = True,
) -> Dict[str, Any]:
    """
    Fit statsmodels MarkovRegression model.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Univariate return series.
    k_regimes : int
        Number of regimes (default 2).
    switching_variance : bool
        Whether residual variance switches across regimes.

    Returns
    -------
    result : Dict[str, Any]
        Fitted model diagnostics, filtered probabilities, and smoothed probabilities.
    """
    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    except ImportError:
        logger.warning("statsmodels is not installed; skipping MarkovRegression comparison.")
        return {
            "available": False,
            "error": "statsmodels is not installed",
        }

    is_series = isinstance(returns, pd.Series)
    index = returns.index if is_series else None
    rets = np.asarray(returns, dtype=np.float64).flatten()

    try:
        model = MarkovRegression(
            rets,
            k_regimes=k_regimes,
            switching_variance=switching_variance,
        )
        fitted = model.fit(disp=False)

        # statsmodels provides filtered_marginal_probabilities and smoothed_marginal_probabilities
        filtered_probs = np.asarray(fitted.filtered_marginal_probabilities)
        smoothed_probs = np.asarray(fitted.smoothed_marginal_probabilities)

        filtered_df = pd.DataFrame(filtered_probs, index=index)
        smoothed_df = pd.DataFrame(smoothed_probs, index=index)

        logger.info(
            f"Fitted statsmodels MarkovRegression (K={k_regimes}, switching_var={switching_variance}): "
            f"LL={fitted.llf:.2f}, AIC={fitted.aic:.2f}, BIC={fitted.bic:.2f}."
        )

        return {
            "available": True,
            "converged": bool(fitted.mle_retvals.get("converged", True)),
            "log_likelihood": float(fitted.llf),
            "aic": float(fitted.aic),
            "bic": float(fitted.bic),
            "k_regimes": k_regimes,
            "switching_variance": switching_variance,
            "filtered_marginal_probabilities": filtered_df,
            "smoothed_marginal_probabilities": smoothed_df,
            "semantics_note": (
                "statsmodels provides both filtered (online) and smoothed (retrospective) posteriors. "
                "Causal online comparisons MUST use filtered_marginal_probabilities."
            ),
        }
    except Exception as e:
        logger.warning(f"statsmodels MarkovRegression fitting failed: {e}")
        return {
            "available": True,
            "converged": False,
            "error": str(e),
        }
