"""
State interpretation, descriptive profiles, dwell times, and assignment entropy.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from regimeflow.models.gaussian_hmm import GaussianHMM


def compute_state_profiles(
    model: GaussianHMM,
    X: np.ndarray,
    gamma: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute descriptive statistics for each latent state:
    - Expected dwell time / persistence
    - State occupancy fraction
    - Feature means and standard deviations
    - Transition matrix diagonal A_kk

    Parameters
    ----------
    model : GaussianHMM
        Fitted HMM model.
    X : np.ndarray of shape (T, d)
        Observation data.
    gamma : np.ndarray of shape (T, K)
        State posterior probabilities.
    feature_names : Optional[List[str]]
        Names of observation features.

    Returns
    -------
    profiles_df : pd.DataFrame
        Table summarizing state traits.
    """
    T, d = X.shape
    K = model.n_states
    f_names = feature_names or [f"feature_{i}" for i in range(d)]

    records = []
    occupancies = np.sum(gamma, axis=0) / T

    for k in range(K):
        # Persistence & Dwell time
        diag_prob = float(model.A_[k, k])
        # Expected duration in first-order Markov chain: 1 / (1 - A_kk)
        expected_dwell = (
            float(1.0 / (1.0 - diag_prob)) if diag_prob < 0.9999 else float("inf")
        )

        rec: Dict[str, Any] = {
            "state_id": k,
            "occupancy_fraction": float(occupancies[k]),
            "persistence_A_kk": diag_prob,
            "expected_dwell_sessions": expected_dwell,
        }

        # Emission means & standard deviations
        for j, fn in enumerate(f_names):
            mean_val = float(model.means_[k, j])
            if model.covariance_type == "full":
                std_val = float(np.sqrt(max(model.covs_[k, j, j], 1e-8)))
            else:
                std_val = float(np.sqrt(max(model.covs_[k, j], 1e-8)))
            rec[f"{fn}_mean"] = mean_val
            rec[f"{fn}_std"] = std_val

        records.append(rec)

    return pd.DataFrame.from_records(records).set_index("state_id")


def compute_posterior_entropy(posteriors: np.ndarray) -> np.ndarray:
    """
    Compute normalized posterior assignment entropy H_t / log(K) in [0, 1].
    Higher entropy indicates greater model uncertainty.
    """
    T, K = posteriors.shape
    if K <= 1:
        return np.zeros(T, dtype=np.float64)

    # H = - sum_k p_k * log(p_k)
    eps = 1e-300
    p = np.maximum(posteriors, eps)
    entropy = -np.sum(p * np.log(p), axis=1)
    normalized_entropy = entropy / np.log(K)
    return np.clip(normalized_entropy, 0.0, 1.0)
