"""
Independent reference comparator using hmmlearn.GaussianHMM.
Used exclusively for algorithmic sanity verification, NOT as a substitute for custom HMM.
"""

from typing import Any, Dict, Optional
import numpy as np
from regimeflow.logging_utils import logger
from regimeflow.models.gaussian_hmm import GaussianHMM


def compare_with_hmmlearn(
    X: np.ndarray,
    n_states: int = 2,
    covariance_type: str = "full",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run comparative analysis between custom GaussianHMM and hmmlearn.GaussianHMM.

    Parameters
    ----------
    X : np.ndarray of shape (T, d)
        Observation sequence.
    n_states : int
        Number of hidden states K.
    covariance_type : str
        Covariance type ("full" or "diag").
    seed : int
        Random seed.

    Returns
    -------
    comparison : Dict[str, Any]
        Dictionary comparing sequence log-likelihoods, fit convergence, and semantics.
    """
    try:
        from hmmlearn import hmm
    except ImportError:
        logger.warning("hmmlearn is not installed; skipping reference comparison.")
        return {
            "available": False,
            "error": "hmmlearn is not installed",
        }

    # 1. Fit custom HMM
    custom_model = GaussianHMM(
        n_states=n_states,
        covariance_type=covariance_type,
        max_iter=100,
        n_restarts=3,
        random_state=seed,
    )
    custom_model.fit(X)
    custom_ll = custom_model.score(X)
    custom_filtered, _ = custom_model.filter(X)
    custom_smoothed = custom_model.smooth(X)

    # 2. Fit hmmlearn reference
    ref_model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=100,
        random_state=seed,
    )
    ref_model.fit(X)
    ref_ll = float(ref_model.score(X))
    ref_posteriors = ref_model.predict_proba(X)

    # Note: hmmlearn.predict_proba returns smoothed posteriors P(z_t | x_{1:T}),
    # which use backward passes, NOT causal forward filtering!
    # Compare ref_posteriors with custom_smoothed (up to permutation)
    # Match states between custom and reference using mean distances
    custom_means = custom_model.means_
    ref_means = ref_model.means_

    # Simple 2-state permutation alignment for reporting
    if n_states == 2:
        dist_direct = np.linalg.norm(custom_means[0] - ref_means[0]) + np.linalg.norm(
            custom_means[1] - ref_means[1]
        )
        dist_inverted = np.linalg.norm(custom_means[0] - ref_means[1]) + np.linalg.norm(
            custom_means[1] - ref_means[0]
        )
        perm = [0, 1] if dist_direct <= dist_inverted else [1, 0]
    else:
        perm = list(range(n_states))

    smoothed_agreement = float(
        np.mean(np.isclose(custom_smoothed[:, perm], ref_posteriors, atol=0.15))
    )

    logger.info(
        f"hmmlearn Comparison: Custom LL={custom_ll:.2f}, hmmlearn LL={ref_ll:.2f}, "
        f"Smoothed posterior agreement={smoothed_agreement * 100:.1f}%."
    )

    return {
        "available": True,
        "n_states": n_states,
        "covariance_type": covariance_type,
        "custom_log_likelihood": float(custom_ll),
        "hmmlearn_log_likelihood": float(ref_ll),
        "likelihood_difference": float(abs(custom_ll - ref_ll)),
        "smoothed_posterior_agreement_fraction": smoothed_agreement,
        "semantics_note": (
            "Confirmed: hmmlearn.predict_proba() is retrospective smoothing (uses future data). "
            "RegimeFlow causal_forward_filter() is strictly forward-only."
        ),
    }
