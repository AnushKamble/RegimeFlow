"""
Tests for BIC model selection, state alignment, and reference hmmlearn comparison.
"""

import numpy as np
import pytest
from regimeflow.config import ModelConfig
from regimeflow.data.synthetic import generate_synthetic_hmm_data
from regimeflow.models.reference_hmmlearn import compare_with_hmmlearn
from regimeflow.models.selection import ModelSelector
from regimeflow.states.alignment import (
    compute_canonical_risk_permutation,
    match_states_hungarian,
    permute_model_parameters,
)


def test_bic_model_selection_identifies_true_k():
    """
    On a clear 2-state synthetic dataset, BIC should penalize over-parameterized models (K=3, 4)
    and select K=2.
    """
    pi = np.array([0.6, 0.4])
    A = np.array([[0.95, 0.05], [0.10, 0.90]])
    means = np.array([[2.5, 2.5], [-2.5, -2.5]])
    covs = np.array([
        [[0.2, 0.0], [0.0, 0.2]],
        [[0.2, 0.0], [0.0, 0.2]],
    ])
    X, _ = generate_synthetic_hmm_data(n_samples=500, pi=pi, A=A, means=means, covs=covs, seed=42)

    config = ModelConfig(
        n_states_candidates=[2, 3, 4],
        covariance_type="full",
        n_restarts=2,
        max_iter=40,
        random_state=42,
    )
    selector = ModelSelector(config)
    best_model, best_res = selector.fit_and_select(X)

    assert best_res.n_states == 2, f"Expected K=2 selected by BIC, got K={best_res.n_states}"
    summary_df = selector.get_summary_dataframe()
    assert len(summary_df) == 3
    assert set(summary_df["n_states"]) == {2, 3, 4}
    assert all(np.isfinite(summary_df["bic"]))
    # Free parameters must strictly increase with K
    p_counts = summary_df.sort_values("n_states")["n_parameters"].tolist()
    assert p_counts == sorted(p_counts) and len(set(p_counts)) == 3


def test_state_alignment_and_permutation():
    means = np.array([[1.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    # State 0 vol = 0.5, State 1 vol = 0.1, State 2 vol = 1.0
    covs = np.array([
        [[0.25, 0.0], [0.0, 0.25]],
        [[0.01, 0.0], [0.0, 0.01]],
        [[1.00, 0.0], [0.0, 1.00]],
    ])

    # Canonical order by volatility should be: State 1 (0.1), State 0 (0.5), State 2 (1.0)
    perm = compute_canonical_risk_permutation(means, covs, "full", vol_feature_idx=0)
    assert np.array_equal(perm, np.array([1, 0, 2]))

    # Test Hungarian matching under exact permutation
    permuted_means = means[perm]
    permuted_covs = covs[perm]
    col_ind = match_states_hungarian(
        ref_means=means,
        ref_covs=covs,
        curr_means=permuted_means,
        curr_covs=permuted_covs,
        covariance_type="full",
    )
    assert len(col_ind) == 3


def test_reference_hmmlearn_comparison():
    X = np.random.default_rng(42).normal(size=(200, 2))
    result = compare_with_hmmlearn(X, n_states=2, covariance_type="full", seed=42)

    assert result["available"] is True
    assert np.isfinite(result["custom_log_likelihood"])
    assert np.isfinite(result["hmmlearn_log_likelihood"])
    # Log likelihoods should be within a reasonable numerical margin
    assert result["likelihood_difference"] < 50.0
    assert "semantics_note" in result
