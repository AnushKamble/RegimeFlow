"""
Circular Moving-Block Bootstrap for paired financial time-series comparisons.
Preserves serial autocorrelation and volatility clustering without IID assumptions.
"""

from typing import Any, Dict
import numpy as np
import pandas as pd


def compute_quick_metrics(rets: np.ndarray) -> Dict[str, float]:
    """Fast metric calculator for bootstrap resampling."""
    N = len(rets)
    if N < 2:
        return {"cagr": 0.0, "sharpe": 0.0, "max_dd": 0.0}

    # Cumulative wealth
    wealth = np.cumprod(1.0 + rets)
    total_ret = wealth[-1]
    cagr = float(total_ret ** (252.0 / N) - 1.0) if total_ret > 0 else -1.0

    vol = float(np.std(rets, ddof=1) * np.sqrt(252.0))
    sharpe = float(cagr / vol) if vol > 1e-8 else 0.0

    running_max = np.maximum.accumulate(wealth)
    dd = (wealth - running_max) / running_max
    max_dd = float(np.min(dd))

    return {"cagr": cagr, "sharpe": sharpe, "max_dd": max_dd}


def paired_block_bootstrap(
    strategy_returns: pd.Series | np.ndarray,
    benchmark_returns: pd.Series | np.ndarray,
    block_length: int = 21,
    n_replications: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Perform circular moving-block bootstrap on paired strategy return differences.

    Parameters
    ----------
    strategy_returns : pd.Series or np.ndarray
        Daily net returns of the candidate strategy.
    benchmark_returns : pd.Series or np.ndarray
        Daily net returns of the benchmark strategy (same timestamps).
    block_length : int
        Block size L in sessions (default 21 sessions ~ 1 calendar month).
    n_replications : int
        Number of bootstrap replications B (default 1000).
    seed : int
        Random seed for bitwise reproducibility.

    Returns
    -------
    bootstrap_results : Dict[str, Any]
        Summary containing empirical differences, 95% confidence intervals, and p-values.
    """
    r_strat = np.asarray(strategy_returns, dtype=np.float64)
    r_bench = np.asarray(benchmark_returns, dtype=np.float64)

    assert len(r_strat) == len(r_bench), "Strategy and benchmark returns must have identical length."
    N = len(r_strat)
    if N < block_length:
        raise ValueError(f"Series length {N} is shorter than bootstrap block length {block_length}.")

    rng = np.random.default_rng(seed)

    # Observed point estimates
    obs_strat = compute_quick_metrics(r_strat)
    obs_bench = compute_quick_metrics(r_bench)
    obs_diff_sharpe = obs_strat["sharpe"] - obs_bench["sharpe"]
    obs_diff_cagr = obs_strat["cagr"] - obs_bench["cagr"]
    obs_diff_max_dd = obs_strat["max_dd"] - obs_bench["max_dd"]

    n_blocks = int(np.ceil(N / block_length))

    diff_sharpes = np.zeros(n_replications, dtype=np.float64)
    diff_cagrs = np.zeros(n_replications, dtype=np.float64)
    diff_max_dds = np.zeros(n_replications, dtype=np.float64)

    # Circular array for boundary wrapping
    strat_circular = np.concatenate([r_strat, r_strat[:block_length]])
    bench_circular = np.concatenate([r_bench, r_bench[:block_length]])

    for b in range(n_replications):
        start_indices = rng.integers(0, N, size=n_blocks)
        sampled_strat = np.concatenate([
            strat_circular[idx : idx + block_length] for idx in start_indices
        ])[:N]
        sampled_bench = np.concatenate([
            bench_circular[idx : idx + block_length] for idx in start_indices
        ])[:N]

        m_strat = compute_quick_metrics(sampled_strat)
        m_bench = compute_quick_metrics(sampled_bench)

        diff_sharpes[b] = m_strat["sharpe"] - m_bench["sharpe"]
        diff_cagrs[b] = m_strat["cagr"] - m_bench["cagr"]
        diff_max_dds[b] = m_strat["max_dd"] - m_bench["max_dd"]

    # Compute empirical percentiles
    def get_ci_and_p(diffs: np.ndarray, obs_diff: float) -> Dict[str, float]:
        q025, q975 = np.percentile(diffs, [2.5, 97.5])
        # Two-sided empirical p-value testing H0: diff == 0
        p_val = 2.0 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
        p_val = min(p_val, 1.0)
        return {
            "observed_diff": obs_diff,
            "ci_lower_95": float(q025),
            "ci_upper_95": float(q975),
            "bootstrap_mean": float(np.mean(diffs)),
            "bootstrap_std": float(np.std(diffs)),
            "p_value": float(p_val),
        }

    return {
        "n_samples": N,
        "block_length": block_length,
        "n_replications": n_replications,
        "sharpe_diff": get_ci_and_p(diff_sharpes, obs_diff_sharpe),
        "cagr_diff": get_ci_and_p(diff_cagrs, obs_diff_cagr),
        "max_dd_diff": get_ci_and_p(diff_max_dds, obs_diff_max_dd),
    }
