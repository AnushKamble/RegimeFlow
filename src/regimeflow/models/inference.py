"""
Vectorized log-domain dynamic programming algorithms for HMM inference.
Implements forward, backward, causal filtering, retrospective smoothing, and Viterbi decoding.
"""

from typing import Tuple
import numpy as np
from scipy.special import logsumexp


def _logsumexp_axis0(a: np.ndarray) -> np.ndarray:
    """Fast numerically stable logsumexp along axis 0."""
    max_a = np.max(a, axis=0)
    return max_a + np.log(np.sum(np.exp(a - max_a), axis=0))


def _logsumexp_axis1(a: np.ndarray) -> np.ndarray:
    """Fast numerically stable logsumexp along axis 1."""
    max_a = np.max(a, axis=1)
    return max_a + np.log(np.sum(np.exp(a - max_a[:, np.newaxis]), axis=1))


def _logsumexp_1d(a: np.ndarray) -> float:
    """Fast numerically stable 1D logsumexp returning float."""
    max_a = np.max(a)
    return float(max_a + np.log(np.sum(np.exp(a - max_a))))


def forward_log_recursion(
    log_pi: np.ndarray,
    log_A: np.ndarray,
    log_B: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Vectorized log-domain forward algorithm.

    Parameters
    ----------
    log_pi : np.ndarray of shape (K,)
        Log initial state distribution.
    log_A : np.ndarray of shape (K, K)
        Log transition matrix: log_A[i, j] = log P(z_t = j | z_{t-1} = i).
    log_B : np.ndarray of shape (T, K)
        Log emission densities: log_B[t, k] = log p(x_t | z_t = k).

    Returns
    -------
    log_alpha : np.ndarray of shape (T, K)
        Log forward variables.
    total_log_lik : float
        Total sequence log-likelihood log p(x_1, ..., x_T).
    """
    T, K = log_B.shape
    log_alpha = np.zeros((T, K), dtype=np.float64)

    # Initial step: log_alpha[0, k] = log_pi[k] + log_B[0, k]
    log_alpha[0] = log_pi + log_B[0]

    # Induction over time; inner state operations are vectorized across K
    for t in range(1, T):
        # log_alpha[t-1, :, None] + log_A has shape (K, K), row i to col k
        # sum over previous state i along axis=0
        v = log_alpha[t - 1, :, None] + log_A
        log_alpha[t] = log_B[t] + _logsumexp_axis0(v)

    total_log_lik = _logsumexp_1d(log_alpha[-1])
    return log_alpha, total_log_lik


def backward_log_recursion(
    log_A: np.ndarray,
    log_B: np.ndarray,
) -> np.ndarray:
    """
    Vectorized log-domain backward algorithm.

    Parameters
    ----------
    log_A : np.ndarray of shape (K, K)
        Log transition matrix.
    log_B : np.ndarray of shape (T, K)
        Log emission densities.

    Returns
    -------
    log_beta : np.ndarray of shape (T, K)
        Log backward variables.
    """
    T, K = log_B.shape
    log_beta = np.zeros((T, K), dtype=np.float64)

    # Base case: log_beta[T-1, :] = 0.0 (log(1.0))
    log_beta[-1] = 0.0

    # Induction backwards; inner state operations are vectorized across K
    for t in range(T - 2, -1, -1):
        # log_A (K, K) + log_B[t+1] (None, K) + log_beta[t+1] (None, K)
        # sum over next state j along axis=1
        v = log_A + log_B[t + 1, None, :] + log_beta[t + 1, None, :]
        log_beta[t] = _logsumexp_axis1(v)

    return log_beta


def causal_forward_filter(
    log_pi: np.ndarray,
    log_A: np.ndarray,
    log_B: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Causal forward filter producing online posterior probabilities P(z_t | x_{1:t})
    and one-step predictive log-likelihoods log p(x_t | x_{1:t-1}).

    Strictly invariant to any future observations after time t.

    Parameters
    ----------
    log_pi : np.ndarray of shape (K,)
        Log initial distribution.
    log_A : np.ndarray of shape (K, K)
        Log transition matrix.
    log_B : np.ndarray of shape (T, K)
        Log emission probabilities through time T.

    Returns
    -------
    filtered_proba : np.ndarray of shape (T, K)
        P(z_t = k | x_{1:t}). Rows sum to 1.
    predictive_ll : np.ndarray of shape (T,)
        One-step ahead predictive log-likelihood log p(x_t | x_{1:t-1}).
    """
    T, K = log_B.shape
    filtered_proba = np.zeros((T, K), dtype=np.float64)
    predictive_ll = np.zeros(T, dtype=np.float64)

    # Time step 0:
    log_prior = log_pi.copy()
    predictive_ll[0] = _logsumexp_1d(log_prior + log_B[0])
    log_posterior = log_prior + log_B[0] - predictive_ll[0]
    post_0 = np.exp(log_posterior)
    filtered_proba[0] = post_0 / np.sum(post_0)

    # Steps 1 to T-1
    for t in range(1, T):
        prev_log_post = np.log(np.maximum(filtered_proba[t - 1], 1e-300))
        # Prior predictive: log P(z_t = k | x_{1:t-1}) = logsumexp_j(prev_post_j + log A_{j,k})
        log_prior_t = _logsumexp_axis0(prev_log_post[:, None] + log_A)

        # One-step predictive density: log p(x_t | x_{1:t-1})
        step_ll = _logsumexp_1d(log_prior_t + log_B[t])
        predictive_ll[t] = step_ll

        # Filtered posterior: log P(z_t = k | x_{1:t})
        log_post_t = log_prior_t + log_B[t] - step_ll
        post_t = np.exp(log_post_t)
        filtered_proba[t] = post_t / np.sum(post_t)

    return filtered_proba, predictive_ll


def compute_smoothed_posteriors(
    log_alpha: np.ndarray,
    log_beta: np.ndarray,
    total_log_lik: float,
) -> np.ndarray:
    """
    Retrospective smoothed state probabilities:
    gamma_t(k) = P(z_t = k | x_{1:T}) = exp(log_alpha_{t,k} + log_beta_{t,k} - log p(X)).

    Explicitly labeled retrospective / offline.
    """
    log_gamma = log_alpha + log_beta - total_log_lik
    gamma = np.exp(log_gamma)
    row_sums = np.sum(gamma, axis=1, keepdims=True)
    return gamma / np.maximum(row_sums, 1e-300)


def compute_transition_posteriors(
    log_alpha: np.ndarray,
    log_beta: np.ndarray,
    log_A: np.ndarray,
    log_B: np.ndarray,
    total_log_lik: float,
) -> np.ndarray:
    """
    Fully vectorized pairwise transition posteriors for EM M-step:
    xi_t(i, j) = P(z_t = i, z_{t+1} = j | x_{1:T}).

    Returns
    -------
    xi : np.ndarray of shape (T - 1, K, K)
    """
    # log_alpha[:-1, :, None] has shape (T-1, K, 1)
    # log_A[None, :, :] has shape (1, K, K)
    # log_B[1:, None, :] has shape (T-1, 1, K)
    # log_beta[1:, None, :] has shape (T-1, 1, K)
    log_xi = (
        log_alpha[:-1, :, None]
        + log_A[None, :, :]
        + log_B[1:, None, :]
        + log_beta[1:, None, :]
        - total_log_lik
    )
    return np.exp(log_xi)


def viterbi_decode(
    log_pi: np.ndarray,
    log_A: np.ndarray,
    log_B: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Vectorized Viterbi dynamic programming decoding for MAP hidden state path:
    z* = argmax_z P(z_1, ..., z_T | x_1, ..., x_T).

    Explicitly labeled retrospective / offline.

    Returns
    -------
    best_path : np.ndarray of shape (T,)
        Optimal hidden state sequence.
    best_score : float
        Log probability of the optimal path.
    """
    T, K = log_B.shape
    log_delta = np.zeros((T, K), dtype=np.float64)
    psi = np.zeros((T, K), dtype=int)

    # Initialization
    log_delta[0] = log_pi + log_B[0]

    # Recursion with vectorized state maximization
    for t in range(1, T):
        # scores has shape (K, K): row j (prev state), col k (curr state)
        scores = log_delta[t - 1, :, None] + log_A
        psi[t] = np.argmax(scores, axis=0)
        log_delta[t] = np.max(scores, axis=0) + log_B[t]

    # Backtracking
    best_path = np.zeros(T, dtype=int)
    best_path[-1] = int(np.argmax(log_delta[-1]))
    best_score = float(log_delta[-1, best_path[-1]])

    for t in range(T - 2, -1, -1):
        best_path[t] = psi[t + 1, best_path[t + 1]]

    return best_path, best_score
