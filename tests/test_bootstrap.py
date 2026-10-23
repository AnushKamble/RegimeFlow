"""
Tests for circular moving-block bootstrap and statistical significance.
"""

import numpy as np
import pytest
from regimeflow.metrics.bootstrap import paired_block_bootstrap


def test_paired_block_bootstrap_reproducibility_and_structure():
    rng = np.random.default_rng(42)
    N = 250
    # Strategy slightly outperforms benchmark
    bench_rets = rng.normal(0.0004, 0.01, size=N)
    strat_rets = bench_rets + rng.normal(0.0002, 0.002, size=N)

    res1 = paired_block_bootstrap(
        strat_rets, bench_rets, block_length=21, n_replications=200, seed=42
    )
    res2 = paired_block_bootstrap(
        strat_rets, bench_rets, block_length=21, n_replications=200, seed=42
    )

    # Bitwise reproducibility with fixed seed
    assert np.isclose(
        res1["sharpe_diff"]["bootstrap_mean"], res2["sharpe_diff"]["bootstrap_mean"]
    )
    assert np.isclose(
        res1["sharpe_diff"]["ci_lower_95"], res2["sharpe_diff"]["ci_lower_95"]
    )
    assert np.isclose(
        res1["sharpe_diff"]["ci_upper_95"], res2["sharpe_diff"]["ci_upper_95"]
    )

    # 95% confidence interval bounds check: lower <= upper
    assert res1["sharpe_diff"]["ci_lower_95"] <= res1["sharpe_diff"]["ci_upper_95"]
    assert res1["cagr_diff"]["ci_lower_95"] <= res1["cagr_diff"]["ci_upper_95"]
    assert 0.0 <= res1["sharpe_diff"]["p_value"] <= 1.0
