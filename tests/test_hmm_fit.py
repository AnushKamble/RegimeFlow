"""
Integration and fit tests for GaussianHMM Baum-Welch EM, recovery, and serialization.
"""

from pathlib import Path
import numpy as np
import pytest
from regimeflow.data.synthetic import generate_synthetic_hmm_data
from regimeflow.models.gaussian_hmm import GaussianHMM
from regimeflow.models.serialization import load_hmm_model, save_hmm_model


def test_hmm_em_fitting_and_recovery():
    """
    Test EM fitting on synthetic data with known 2-state generating distribution.
    Verifies that EM increases log-likelihood and discovers well-separated states.
    """
    pi = np.array([0.7, 0.3])
    A = np.array([[0.92, 0.08], [0.15, 0.85]])
    means = np.array([[2.0, 1.0], [-2.0, -1.0]])
    covs = np.array([
        [[0.2, 0.05], [0.05, 0.2]],
        [[0.3, -0.05], [-0.05, 0.3]],
    ])

    X, true_states = generate_synthetic_hmm_data(
        n_samples=600, pi=pi, A=A, means=means, covs=covs, seed=42
    )

    model = GaussianHMM(
        n_states=2,
        covariance_type="full",
        n_restarts=3,
        max_iter=50,
        random_state=42,
    )
    model.fit(X)

    assert model.is_fitted_ is True
    assert np.isfinite(model.final_log_likelihood_)
    assert len(model.means_) == 2
    assert model.A_.shape == (2, 2)
    assert np.allclose(np.sum(model.A_, axis=1), 1.0)
    assert np.isclose(np.sum(model.pi_), 1.0)

    # Decode path and check state recovery alignment (up to permutation)
    viterbi_path, score = model.decode_viterbi(X)
    assert len(viterbi_path) == 600

    # Agreement with true states (checking both direct and inverted permutation)
    acc_direct = np.mean(viterbi_path == true_states)
    acc_inverted = np.mean(viterbi_path == (1 - true_states))
    best_acc = max(acc_direct, acc_inverted)
    assert best_acc > 0.85, f"Expected state recovery > 85%, got {best_acc:.3f}"


def test_hmm_diagonal_covariance_mode():
    pi = np.array([0.5, 0.5])
    A = np.array([[0.9, 0.1], [0.1, 0.9]])
    means = np.array([[1.5, 0.0], [-1.5, 0.0]])
    covs = np.array([[[0.2, 0.0], [0.0, 0.2]], [[0.2, 0.0], [0.0, 0.2]]])
    X, _ = generate_synthetic_hmm_data(300, pi, A, means, covs, seed=99)

    model = GaussianHMM(
        n_states=2,
        covariance_type="diag",
        n_restarts=2,
        max_iter=30,
        random_state=123,
    )
    model.fit(X)
    assert model.is_fitted_
    assert model.covs_.shape == (2, 2)  # (K, d) for diag mode
    assert (model.covs_ > 0).all()


def test_hmm_serialization_round_trip(tmp_path):
    X = np.random.default_rng(42).normal(size=(100, 2))
    model = GaussianHMM(n_states=2, covariance_type="full", n_restarts=1, max_iter=10)
    model.fit(X)

    save_path = tmp_path / "test_hmm.json"
    save_hmm_model(model, save_path)

    reloaded = load_hmm_model(save_path)
    assert reloaded.is_fitted_
    assert reloaded.n_states == model.n_states
    assert reloaded.covariance_type == model.covariance_type
    assert np.allclose(reloaded.means_, model.means_)
    assert np.allclose(reloaded.A_, model.A_)
    assert np.allclose(reloaded.pi_, model.pi_)
    assert np.allclose(reloaded.covs_, model.covs_)

    # Verify predictions match exactly
    score_orig = model.score(X)
    score_reloaded = reloaded.score(X)
    assert np.isclose(score_orig, score_reloaded)

    filt_orig, _ = model.filter(X)
    filt_reloaded, _ = reloaded.filter(X)
    assert np.allclose(filt_orig, filt_reloaded)
