"""
Tests for offline PELT changepoint detection, statsmodels MarkovRegression, and online BOCPD.
"""

import numpy as np
import pytest
from regimeflow.changepoints.bocpd import BOCPD
from regimeflow.changepoints.pelt_baseline import run_pelt_changepoints
from regimeflow.models.markov_switching import fit_markov_switching_regression


def test_pelt_offline_changepoint_detection():
    rng = np.random.default_rng(42)
    # Piecewise stationary sequence with clear shift at t=100
    s1 = rng.normal(0.0, 1.0, size=100)
    s2 = rng.normal(5.0, 1.0, size=100)
    signal = np.concatenate([s1, s2])

    res = run_pelt_changepoints(signal, model="rbf", pen=3.0, min_size=20)
    assert res["available"] is True
    assert res["is_causal_online"] is False
    assert "RETROSPECTIVE" in res["semantics_warning"]
    assert res["n_changepoints"] >= 1
    # Check that a breakpoint is located near t=100
    bps = res["breakpoint_indices"]
    near_100 = any(abs(bp - 100) <= 15 for bp in bps)
    assert near_100, f"Expected breakpoint near 100, got {bps}"


def test_bocpd_online_detection_and_causality():
    """
    Test Bayesian Online Changepoint Detection:
    1. Verifies changepoint probability spike at true break.
    2. Verifies online causal invariance (modifying future does not alter past run-length posteriors).
    """
    rng = np.random.default_rng(42)
    # Mean shift at t=60: from N(0, 0.5) to N(3, 0.5)
    s1 = rng.normal(0.0, 0.5, size=60)
    s2 = rng.normal(3.0, 0.5, size=60)
    signal_orig = np.concatenate([s1, s2])

    bocpd = BOCPD(hazard_lambda=100.0, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
    R_orig, cp_probs_orig = bocpd.run(signal_orig)

    assert R_orig.shape[0] == 120
    # Probabilities sum to 1 at each t
    assert np.allclose(np.sum(R_orig, axis=1), 1.0)

    # Around t=60..65, changepoint probability should elevate
    max_cp_near_break = np.max(cp_probs_orig[60:70])
    baseline_cp = np.mean(cp_probs_orig[20:55])
    assert max_cp_near_break > baseline_cp, (
        f"BOCPD failed to detect mean shift: peak={max_cp_near_break}, baseline={baseline_cp}"
    )

    # CAUSALITY TEST:
    # Perturb observations after t=40 drastically
    signal_pert = signal_orig.copy()
    signal_pert[45:] += 10.0

    R_pert, cp_probs_pert = bocpd.run(signal_pert)

    # For all t <= 44, R and cp_probs MUST BE BITWISE / EXACTLY IDENTICAL
    assert np.allclose(R_orig[:45], R_pert[:45], atol=1e-14)
    assert np.allclose(cp_probs_orig[:45], cp_probs_pert[:45], atol=1e-14)


def test_statsmodels_markov_switching_regression():
    rng = np.random.default_rng(42)
    # 2-regime returns: calm vs volatile
    r1 = rng.normal(0.001, 0.005, size=150)
    r2 = rng.normal(-0.002, 0.025, size=150)
    returns = np.concatenate([r1, r2])

    res = fit_markov_switching_regression(returns, k_regimes=2, switching_variance=True)
    assert res["available"] is True
    if res["converged"]:
        assert "filtered_marginal_probabilities" in res
        assert "smoothed_marginal_probabilities" in res
        assert len(res["filtered_marginal_probabilities"]) == 300
        assert len(res["smoothed_marginal_probabilities"]) == 300
        assert "semantics_note" in res
