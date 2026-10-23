"""
Custom Gaussian HMM modeling, inference, and serialization.
"""

from regimeflow.models.gaussian_emissions import (
    compute_gaussian_log_emissions,
    regularize_covariance,
)
from regimeflow.models.gaussian_hmm import GaussianHMM
from regimeflow.models.inference import (
    backward_log_recursion,
    causal_forward_filter,
    compute_smoothed_posteriors,
    compute_transition_posteriors,
    forward_log_recursion,
    viterbi_decode,
)
from regimeflow.models.initialization import initialize_parameters
from regimeflow.models.serialization import load_hmm_model, save_hmm_model

__all__ = [
    "GaussianHMM",
    "compute_gaussian_log_emissions",
    "regularize_covariance",
    "forward_log_recursion",
    "backward_log_recursion",
    "causal_forward_filter",
    "compute_smoothed_posteriors",
    "compute_transition_posteriors",
    "viterbi_decode",
    "initialize_parameters",
    "save_hmm_model",
    "load_hmm_model",
]
