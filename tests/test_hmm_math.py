"""
Exhaustive mathematical verification of HMM algorithms against brute-force path enumeration.
"""

import itertools
import numpy as np
import pytest
from scipy.special import logsumexp
from regimeflow.models.gaussian_emissions import compute_gaussian_log_emissions
from regimeflow.models.gaussian_hmm import GaussianHMM
from regimeflow.models.inference import (
    backward_log_recursion,
    causal_forward_filter,
    compute_smoothed_posteriors,
    forward_log_recursion,
    viterbi_decode,
)


def test_brute_force_path_enumeration_comparison():
    """
    CRITICAL MATH PROOF:
    On a tiny model (K=2, T=3, d=1), enumerate all K^T = 8 possible hidden paths
    and verify that forward algorithm, smoothing, and Viterbi match exact brute-force values.
    """
    K = 2
    T = 3
    d = 1

    pi = np.array([0.65, 0.35], dtype=np.float64)
    A = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)
    means = np.array([[1.0], [-1.0]], dtype=np.float64)
    covs = np.array([[[0.25]], [[0.5]]], dtype=np.float64)

    X = np.array([[0.8], [0.2], [-0.9]], dtype=np.float64)

    # 1. Compute emission probabilities
    log_B = compute_gaussian_log_emissions(X, means, covs, "full")
    B = np.exp(log_B)

    # 2. Brute-force enumeration of all 8 paths
    all_paths = list(itertools.product(range(K), repeat=T))
    path_joint_probs = []

    for path in all_paths:
        # P(z_0, z_1, z_2, x_0, x_1, x_2) = pi[z_0] * B[0, z_0] * A[z_0, z_1] * B[1, z_1] * A[z_1, z_2] * B[2, z_2]
        p = (
            pi[path[0]]
            * B[0, path[0]]
            * A[path[0], path[1]]
            * B[1, path[1]]
            * A[path[1], path[2]]
            * B[2, path[2]]
        )
        path_joint_probs.append((path, p))

    brute_total_prob = sum(p for _, p in path_joint_probs)
    brute_total_log_lik = np.log(brute_total_prob)

    # Brute-force smoothed marginals gamma_t(k)
    brute_gamma = np.zeros((T, K), dtype=np.float64)
    for t in range(T):
        for k in range(K):
            brute_gamma[t, k] = (
                sum(p for path, p in path_joint_probs if path[t] == k) / brute_total_prob
            )

    # Brute-force Viterbi MAP path
    best_path_tuple, best_joint_p = max(path_joint_probs, key=lambda item: item[1])
    brute_best_path = np.array(best_path_tuple)
    brute_best_log_score = np.log(best_joint_p)

    # 3. Dynamic Programming execution
    log_pi = np.log(pi)
    log_A = np.log(A)

    log_alpha, dp_total_ll = forward_log_recursion(log_pi, log_A, log_B)
    log_beta = backward_log_recursion(log_A, log_B)
    dp_gamma = compute_smoothed_posteriors(log_alpha, log_beta, dp_total_ll)
    dp_viterbi_path, dp_viterbi_score = viterbi_decode(log_pi, log_A, log_B)

    # 4. Assert exact agreement
    assert np.isclose(dp_total_ll, brute_total_log_lik, atol=1e-12), (
        f"Forward likelihood {dp_total_ll} != brute force {brute_total_log_lik}"
    )
    assert np.allclose(dp_gamma, brute_gamma, atol=1e-12), (
        f"Smoothed gamma mismatch: DP={dp_gamma}, Brute={brute_gamma}"
    )
    assert np.array_equal(dp_viterbi_path, brute_best_path), (
        f"Viterbi path mismatch: DP={dp_viterbi_path}, Brute={brute_best_path}"
    )
    assert np.isclose(dp_viterbi_score, brute_best_log_score, atol=1e-12)


def test_causal_filtering_vs_retrospective_smoothing_semantics():
    """
    Asserts the crucial temporal distinction:
    - Modifying observation at T does NOT alter causal filtered probabilities at t < T.
    - Modifying observation at T DOES alter retrospective smoothed probabilities at t < T.
    """
    pi = np.array([0.5, 0.5])
    A = np.array([[0.7, 0.3], [0.3, 0.7]])
    means = np.array([[1.0], [-1.0]])
    covs = np.array([[[1.0]], [[1.0]]])

    # t=0 is 0.5, t=1 is 0.0 (ambiguous point halfway between means)
    X_orig = np.array([[0.5], [0.0], [2.0]])
    log_B_orig = compute_gaussian_log_emissions(X_orig, means, covs, "full")
    log_pi = np.log(pi)
    log_A = np.log(A)

    filt_orig, _ = causal_forward_filter(log_pi, log_A, log_B_orig)
    log_alpha_orig, ll_orig = forward_log_recursion(log_pi, log_A, log_B_orig)
    log_beta_orig = backward_log_recursion(log_A, log_B_orig)
    smooth_orig = compute_smoothed_posteriors(log_alpha_orig, log_beta_orig, ll_orig)

    # In X_pert, final observation at T=2 flips from +2.0 to -2.0
    X_pert = np.array([[0.5], [0.0], [-2.0]])
    log_B_pert = compute_gaussian_log_emissions(X_pert, means, covs, "full")

    filt_pert, _ = causal_forward_filter(log_pi, log_A, log_B_pert)
    log_alpha_pert, ll_pert = forward_log_recursion(log_pi, log_A, log_B_pert)
    log_beta_pert = backward_log_recursion(log_A, log_B_pert)
    smooth_pert = compute_smoothed_posteriors(log_alpha_pert, log_beta_pert, ll_pert)

    # 1. Causal filter at t=0 and t=1 MUST BE EXACTLY IDENTICAL
    assert np.allclose(filt_orig[0], filt_pert[0], atol=1e-14)
    assert np.allclose(filt_orig[1], filt_pert[1], atol=1e-14)

    # 2. Retrospective smoothed posterior at t=1 MUST DIFFER significantly
    # Because future observation at t=2 (+2 vs -2) pulls the retrospective belief
    assert abs(smooth_orig[1, 0] - smooth_pert[1, 0]) > 0.05, (
        f"Smoothed posterior at t=1 did not change: {smooth_orig[1, 0]} vs {smooth_pert[1, 0]}"
    )


def test_parameter_count_formulas():
    """Verify free parameter counting for BIC selection."""
    model_full = GaussianHMM(n_states=3, covariance_type="full")
    model_full.n_features_ = 4
    # K=3, d=4
    # p_init = 3 - 1 = 2
    # p_trans = 3 * (3 - 1) = 6
    # p_means = 3 * 4 = 12
    # p_cov_full = 3 * 4 * 5 / 2 = 30
    # total = 2 + 6 + 12 + 30 = 50
    assert model_full.n_parameters == 50

    model_diag = GaussianHMM(n_states=3, covariance_type="diag")
    model_diag.n_features_ = 4
    # p_cov_diag = 3 * 4 = 12
    # total = 2 + 6 + 12 + 12 = 32
    assert model_diag.n_parameters == 32
