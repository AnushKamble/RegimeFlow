"""
Multivariate Gaussian emission log-densities with stable Cholesky algebra
and controlled positive-definite regularization.
"""

from typing import Literal, Tuple
import numpy as np
import scipy.linalg


def regularize_covariance(
    cov: np.ndarray,
    reg_factor: float = 1e-4,
    max_jitter: float = 1e-2,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Ensure covariance matrix is symmetric positive definite via Cholesky decomposition
    and adaptive diagonal jitter escalation.

    Parameters
    ----------
    cov : np.ndarray of shape (d, d)
        Input covariance matrix.
    reg_factor : float
        Initial diagonal floor added to eigenvalues.
    max_jitter : float
        Maximum allowable jitter before declaring singularity.

    Returns
    -------
    cov_reg : np.ndarray of shape (d, d)
        Symmetric positive-definite regularized covariance.
    L : np.ndarray of shape (d, d)
        Lower Cholesky factor such that L @ L.T = cov_reg.
    log_det : float
        Log determinant of cov_reg.

    Raises
    ------
    FloatingPointError
        If covariance cannot be factored even after jitter escalation.
    """
    d = cov.shape[0]
    # Symmetrize
    cov_sym = 0.5 * (cov + cov.T)
    jitter = reg_factor

    while jitter <= max_jitter:
        try:
            cov_reg = cov_sym + jitter * np.eye(d)
            L = scipy.linalg.cholesky(cov_reg, lower=True)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            return cov_reg, L, log_det
        except (scipy.linalg.LinAlgError, ValueError):
            jitter *= 10.0

    # If Cholesky still fails, perform eigenvalue clipping
    eigvals, eigvecs = np.linalg.eigh(cov_sym)
    eigvals_clipped = np.maximum(eigvals, reg_factor)
    cov_reconstructed = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    cov_reconstructed = 0.5 * (cov_reconstructed + cov_reconstructed.T)
    L = scipy.linalg.cholesky(cov_reconstructed, lower=True)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return cov_reconstructed, L, log_det


def compute_gaussian_log_emissions(
    X: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    covariance_type: Literal["full", "diag"] = "full",
    cov_regularization: float = 1e-4,
) -> np.ndarray:
    """
    Compute log p(x_t | z_t = k) for all t=1..T and k=1..K.

    Parameters
    ----------
    X : np.ndarray of shape (T, d)
        Observation sequence.
    means : np.ndarray of shape (K, d)
        State emission means.
    covs : np.ndarray of shape (K, d, d) if full, or (K, d) if diag
        State emission covariances.
    covariance_type : "full" or "diag"
        Covariance structure.
    cov_regularization : float
        Diagonal regularizer.

    Returns
    -------
    log_B : np.ndarray of shape (T, K)
        log_B[t, k] = log N(x_t | means[k], covs[k])
    """
    T, d = X.shape
    K = means.shape[0]
    log_B = np.zeros((T, K), dtype=np.float64)

    const_term = -0.5 * d * np.log(2.0 * np.pi)

    if covariance_type == "full":
        for k in range(K):
            _, L, log_det = regularize_covariance(
                covs[k], reg_factor=cov_regularization
            )
            diff = X - means[k]  # shape (T, d)
            # Solve L @ Y = diff.T  -> Y has shape (d, T)
            # Solves triangular system without computing inverse
            Y = scipy.linalg.solve_triangular(L, diff.T, lower=True)
            # Mahalanobis distance for each t is sum of squares of column t
            mahalanobis = np.sum(Y**2, axis=0)  # shape (T,)
            log_B[:, k] = const_term - 0.5 * log_det - 0.5 * mahalanobis

    elif covariance_type == "diag":
        for k in range(K):
            var_k = np.maximum(covs[k], cov_regularization)
            log_det = np.sum(np.log(var_k))
            diff = X - means[k]  # shape (T, d)
            mahalanobis = np.sum((diff**2) / var_k, axis=1)  # shape (T,)
            log_B[:, k] = const_term - 0.5 * log_det - 0.5 * mahalanobis
    else:
        raise ValueError(f"Unknown covariance_type '{covariance_type}'. Must be 'full' or 'diag'.")

    return log_B
