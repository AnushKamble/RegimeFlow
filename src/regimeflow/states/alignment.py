"""
State label alignment across folds and canonical risk-ordered permutations.
Prevents arbitrary state integer ID swapping from corrupting tracking.
"""

from typing import Literal, Tuple
import numpy as np
import scipy.optimize


def get_state_volatilities(
    covs: np.ndarray,
    covariance_type: Literal["full", "diag"] = "full",
    vol_feature_idx: int = 0,
) -> np.ndarray:
    """
    Extract state-specific volatility proxy for state ranking.
    """
    K = covs.shape[0]
    vols = np.zeros(K, dtype=np.float64)
    for k in range(K):
        if covariance_type == "full":
            var_val = covs[k, vol_feature_idx, vol_feature_idx]
        else:
            var_val = covs[k, vol_feature_idx]
        vols[k] = np.sqrt(max(var_val, 1e-8))
    return vols


def compute_canonical_risk_permutation(
    means: np.ndarray,
    covs: np.ndarray,
    covariance_type: Literal["full", "diag"] = "full",
    vol_feature_idx: int = 0,
) -> np.ndarray:
    """
    Compute canonical permutation sorting states by ascending volatility:
    State 0 = Lowest Volatility (Calm) ... State K-1 = Highest Volatility (Crisis/Stress).

    Returns
    -------
    perm : np.ndarray of shape (K,)
        Permutation mapping: canonical_k = perm[k].
    """
    vols = get_state_volatilities(covs, covariance_type, vol_feature_idx)
    return np.argsort(vols)


def match_states_hungarian(
    ref_means: np.ndarray,
    ref_covs: np.ndarray,
    curr_means: np.ndarray,
    curr_covs: np.ndarray,
    covariance_type: Literal["full", "diag"] = "full",
) -> np.ndarray:
    """
    Match current state identities to reference fold state identities using
    the Hungarian bipartite matching algorithm on distribution distances.

    Parameters
    ----------
    ref_means : (K, d)
    ref_covs : (K, d, d) or (K, d)
    curr_means : (K, d)
    curr_covs : (K, d, d) or (K, d)

    Returns
    -------
    col_ind : np.ndarray of shape (K,)
        Mapping such that curr_state = col_ind[ref_state].
    """
    K = ref_means.shape[0]
    cost_matrix = np.zeros((K, K), dtype=np.float64)

    ref_vols = get_state_volatilities(ref_covs, covariance_type)
    curr_vols = get_state_volatilities(curr_covs, covariance_type)

    for i in range(K):
        for j in range(K):
            mean_dist = np.linalg.norm(ref_means[i] - curr_means[j])
            vol_dist = abs(ref_vols[i] - curr_vols[j])
            cost_matrix[i, j] = mean_dist + 2.0 * vol_dist

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    return col_ind


def permute_model_parameters(
    pi: np.ndarray,
    A: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    perm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply state permutation to HMM parameters.
    """
    pi_perm = pi[perm].copy()
    A_perm = A[perm][:, perm].copy()
    means_perm = means[perm].copy()
    covs_perm = covs[perm].copy()

    # Re-verify row stochasticity
    pi_perm /= np.sum(pi_perm)
    A_perm /= np.sum(A_perm, axis=1, keepdims=True)

    return pi_perm, A_perm, means_perm, covs_perm
