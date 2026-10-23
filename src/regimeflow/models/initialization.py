"""
Reproducible parameter initialization for Gaussian HMMs.
"""

from typing import Literal, Tuple
import numpy as np


def initialize_parameters(
    X: np.ndarray,
    n_states: int,
    covariance_type: Literal["full", "diag"] = "full",
    seed: int = 42,
    cov_regularization: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize (pi, A, means, covs) using seeded quantile partitioning.

    Parameters
    ----------
    X : np.ndarray of shape (T, d)
        Observation data.
    n_states : int
        Number of states K.
    covariance_type : "full" or "diag"
        Covariance parameterization.
    seed : int
        Random seed.
    cov_regularization : float
        Diagonal floor.

    Returns
    -------
    pi : np.ndarray of shape (K,)
    A : np.ndarray of shape (K, K)
    means : np.ndarray of shape (K, d)
    covs : np.ndarray of shape (K, d, d) if full, or (K, d) if diag
    """
    rng = np.random.default_rng(seed)
    T, d = X.shape
    K = n_states

    # 1. Initial state distribution pi: slightly perturbed uniform
    pi_noise = rng.uniform(0.9, 1.1, size=K)
    pi = pi_noise / np.sum(pi_noise)

    # 2. Transition matrix A: strong diagonal persistence (0.85 to 0.95)
    A = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        diag_val = rng.uniform(0.85, 0.95)
        off_diag = (1.0 - diag_val) / max(K - 1, 1)
        A[i, :] = off_diag
        A[i, i] = diag_val
        # add tiny positive noise to avoid hard zeros
        A[i, :] += rng.uniform(0.001, 0.005, size=K)
        A[i, :] /= np.sum(A[i, :])

    # 3. Means & Covariances: Quantile partitioning along primary variance axis
    # Project data onto direction of maximum variance
    X_centered = X - np.mean(X, axis=0)
    # 1st principal component
    try:
        _, _, vh = np.linalg.svd(X_centered, full_matrices=False)
        scores = X_centered @ vh[0]
    except Exception:
        scores = X[:, 0]

    # Assign points to quantiles
    quantiles = np.linspace(0, 100, K + 1)
    bin_edges = np.percentile(scores, quantiles)
    bin_indices = np.digitize(scores, bin_edges[1:-1])

    means = np.zeros((K, d), dtype=np.float64)
    if covariance_type == "full":
        covs = np.zeros((K, d, d), dtype=np.float64)
    else:
        covs = np.zeros((K, d), dtype=np.float64)

    global_cov = np.cov(X, rowvar=False) if T > 1 else np.eye(d)
    if global_cov.ndim == 0:
        global_cov = np.array([[global_cov]])

    for k in range(K):
        cluster_points = X[bin_indices == k]
        if len(cluster_points) > d:
            means[k] = np.mean(cluster_points, axis=0)
            if covariance_type == "full":
                c = np.cov(cluster_points, rowvar=False) + cov_regularization * np.eye(d)
                covs[k] = 0.5 * (c + c.T)
            else:
                covs[k] = np.var(cluster_points, axis=0, ddof=1) + cov_regularization
        else:
            # Fallback for empty/sparse bin: global mean + random jitter
            jitter = rng.normal(0, 0.1, size=d)
            means[k] = np.mean(X, axis=0) + jitter
            if covariance_type == "full":
                covs[k] = global_cov + cov_regularization * np.eye(d)
            else:
                covs[k] = np.var(X, axis=0) + cov_regularization

    return pi, A, means, covs
