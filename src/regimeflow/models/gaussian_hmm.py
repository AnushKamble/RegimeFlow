"""
Multivariate Gaussian Hidden Markov Model implemented from first principles.
Includes log-domain Baum-Welch EM, causal filtering, smoothing, and Viterbi decoding.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
import scipy.linalg
from regimeflow.logging_utils import logger
from regimeflow.models.gaussian_emissions import (
    compute_gaussian_log_emissions,
    regularize_covariance,
)
from regimeflow.models.inference import (
    backward_log_recursion,
    causal_forward_filter,
    compute_smoothed_posteriors,
    compute_transition_posteriors,
    forward_log_recursion,
    viterbi_decode,
)
from regimeflow.models.initialization import initialize_parameters


class GaussianHMM:
    """
    Multivariate Gaussian Hidden Markov Model.

    Parameters
    ----------
    n_states : int
        Number of hidden states K (default 2).
    covariance_type : "full" or "diag"
        Covariance matrix parameterization.
    max_iter : int
        Maximum EM iterations per restart.
    tol : float
        Log-likelihood improvement convergence tolerance.
    n_restarts : int
        Number of seeded initializations to run.
    cov_regularization : float
        Diagonal regularizer floor for covariance matrices.
    min_effective_occupancy : float
        Minimum fraction of total observations a state must claim to avoid degeneracy.
    random_state : int
        Base random seed.
    """

    def __init__(
        self,
        n_states: int = 2,
        covariance_type: Literal["full", "diag"] = "full",
        max_iter: int = 100,
        tol: float = 1e-4,
        n_restarts: int = 5,
        cov_regularization: float = 1e-4,
        min_effective_occupancy: float = 0.02,
        random_state: int = 42,
    ):
        if n_states < 2:
            raise ValueError(f"n_states must be >= 2, got {n_states}")
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.n_restarts = n_restarts
        self.cov_regularization = cov_regularization
        self.min_effective_occupancy = min_effective_occupancy
        self.random_state = random_state

        # Fitted parameters
        self.pi_: Optional[np.ndarray] = None
        self.A_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.covs_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.is_fitted_: bool = False

        # Fit metadata & restart diagnostics
        self.final_log_likelihood_: float = float("-inf")
        self.n_iter_: int = 0
        self.restarts_log_: List[Dict[str, Any]] = []

    @property
    def n_parameters(self) -> int:
        """
        Nominal free parameter count for model selection.
        - Initial distribution: K - 1
        - Transition matrix: K * (K - 1)
        - Means: K * d
        - Covariances:
            Full: K * d * (d + 1) / 2
            Diag: K * d
        """
        if self.n_features_ is None:
            raise ValueError("Model must be fit to calculate parameter count.")
        K = self.n_states
        d = self.n_features_

        p_init = K - 1
        p_trans = K * (K - 1)
        p_means = K * d
        if self.covariance_type == "full":
            p_cov = K * d * (d + 1) // 2
        else:
            p_cov = K * d

        return p_init + p_trans + p_means + p_cov

    def compute_bic(self, n_samples: int, log_likelihood: Optional[float] = None) -> float:
        """
        Calculate Bayesian Information Criterion: BIC = -2 * log_L + p * log(N).
        """
        ll = self.final_log_likelihood_ if log_likelihood is None else log_likelihood
        p = self.n_parameters
        return -2.0 * ll + p * np.log(n_samples)

    def compute_aic(self, log_likelihood: Optional[float] = None) -> float:
        """
        Calculate Akaike Information Criterion: AIC = -2 * log_L + 2 * p.
        """
        ll = self.final_log_likelihood_ if log_likelihood is None else log_likelihood
        p = self.n_parameters
        return -2.0 * ll + 2.0 * p

    def fit(self, X: np.ndarray) -> "GaussianHMM":
        """
        Fit model using multi-start Expectation-Maximization (Baum-Welch).
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        T, d = X.shape
        self.n_features_ = d

        best_ll = float("-inf")
        best_params = None
        self.restarts_log_ = []

        for r in range(self.n_restarts):
            seed = self.random_state + r * 101
            restart_record: Dict[str, Any] = {
                "restart_idx": r,
                "seed": seed,
                "converged": False,
                "degenerate": False,
                "iterations": 0,
                "final_ll": float("-inf"),
                "reason": "Unknown",
            }

            try:
                # Initialize parameters
                pi, A, means, covs = initialize_parameters(
                    X,
                    self.n_states,
                    covariance_type=self.covariance_type,
                    seed=seed,
                    cov_regularization=self.cov_regularization,
                )

                prev_ll = float("-inf")
                converged = False

                for iteration in range(1, self.max_iter + 1):
                    # 1. Emission log probabilities
                    log_B = compute_gaussian_log_emissions(
                        X, means, covs, self.covariance_type, self.cov_regularization
                    )

                    # 2. E-step
                    log_pi = np.log(np.maximum(pi, 1e-300))
                    log_A = np.log(np.maximum(A, 1e-300))

                    log_alpha, current_ll = forward_log_recursion(log_pi, log_A, log_B)
                    log_beta = backward_log_recursion(log_A, log_B)

                    gamma = compute_smoothed_posteriors(log_alpha, log_beta, current_ll)
                    xi = compute_transition_posteriors(
                        log_alpha, log_beta, log_A, log_B, current_ll
                    )

                    # Occupancy check
                    occupancies = np.sum(gamma, axis=0)  # shape (K,)
                    min_occupancy_frac = np.min(occupancies) / T
                    if min_occupancy_frac < self.min_effective_occupancy:
                        restart_record["degenerate"] = True
                        restart_record["reason"] = (
                            f"State collapsed: min occupancy fraction {min_occupancy_frac:.4f} "
                            f"< {self.min_effective_occupancy}"
                        )
                        break

                    # Check convergence
                    if iteration > 1 and abs(current_ll - prev_ll) < self.tol:
                        converged = True
                        restart_record["iterations"] = iteration
                        restart_record["converged"] = True
                        restart_record["final_ll"] = current_ll
                        restart_record["reason"] = "Log-likelihood tolerance reached"
                        break

                    prev_ll = current_ll

                    # 3. M-step
                    # Initial state distribution update
                    pi = gamma[0] / np.sum(gamma[0])

                    # Transition matrix update
                    # sum over t=0..T-2 of xi_t(i, j) / sum over t=0..T-2 of gamma_t(i)
                    sum_xi = np.sum(xi, axis=0)  # (K, K)
                    sum_gamma_trans = np.sum(gamma[:-1], axis=0, keepdims=True).T  # (K, 1)
                    A_new = (sum_xi + 1e-8) / np.maximum(sum_gamma_trans + 1e-8 * self.n_states, 1e-300)
                    A = A_new / np.sum(A_new, axis=1, keepdims=True)

                    # Means update: sum_t gamma_t(k) * x_t / sum_t gamma_t(k)
                    means = (gamma.T @ X) / np.maximum(occupancies[:, np.newaxis], 1e-300)

                    # Covariances update
                    if self.covariance_type == "full":
                        for k in range(self.n_states):
                            diff = X - means[k]
                            weighted_diff = diff * gamma[:, k : k + 1]
                            c = (weighted_diff.T @ diff) / max(occupancies[k], 1e-300)
                            c = 0.5 * (c + c.T) + self.cov_regularization * np.eye(d)
                            covs[k] = c
                    else:
                        for k in range(self.n_states):
                            diff_sq = (X - means[k]) ** 2
                            weighted_sq = np.sum(diff_sq * gamma[:, k : k + 1], axis=0)
                            covs[k] = (weighted_sq / max(occupancies[k], 1e-300)) + self.cov_regularization

                if not converged and not restart_record["degenerate"]:
                    restart_record["iterations"] = self.max_iter
                    restart_record["final_ll"] = prev_ll
                    restart_record["reason"] = "Reached max iterations"

                # Check if this valid restart is best
                if not restart_record["degenerate"] and restart_record["final_ll"] > best_ll:
                    best_ll = restart_record["final_ll"]
                    best_params = (pi.copy(), A.copy(), means.copy(), covs.copy())
                    self.n_iter_ = restart_record["iterations"]

            except Exception as e:
                restart_record["reason"] = f"Numerical/EM failure: {e}"

            self.restarts_log_.append(restart_record)

        if best_params is None:
            raise RuntimeError(
                f"All {self.n_restarts} EM restarts failed for K={self.n_states}. "
                f"Diagnostics: {[r['reason'] for r in self.restarts_log_]}"
            )

        self.pi_, self.A_, self.means_, self.covs_ = best_params
        self.final_log_likelihood_ = best_ll
        self.is_fitted_ = True
        logger.info(
            f"Fitted GaussianHMM (K={self.n_states}, {self.covariance_type}): "
            f"LL={best_ll:.2f}, BIC={self.compute_bic(T):.2f}, "
            f"restarts_valid={sum(not r['degenerate'] for r in self.restarts_log_)}/{self.n_restarts}"
        )
        return self

    def score(self, X: np.ndarray) -> float:
        """Calculate total sequence log-likelihood."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fit before scoring.")
        X = np.asarray(X, dtype=np.float64)
        log_B = compute_gaussian_log_emissions(
            X, self.means_, self.covs_, self.covariance_type, self.cov_regularization
        )
        log_pi = np.log(np.maximum(self.pi_, 1e-300))
        log_A = np.log(np.maximum(self.A_, 1e-300))
        _, total_ll = forward_log_recursion(log_pi, log_A, log_B)
        return total_ll

    def filter(
        self,
        X: np.ndarray,
        initial_prior: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute causal filtered posterior probabilities P(z_t | x_{1:t}) and
        one-step predictive log-likelihoods.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fit before filtering.")
        X = np.asarray(X, dtype=np.float64)
        log_B = compute_gaussian_log_emissions(
            X, self.means_, self.covs_, self.covariance_type, self.cov_regularization
        )
        init_pi = self.pi_ if initial_prior is None else initial_prior
        log_pi = np.log(np.maximum(init_pi, 1e-300))
        log_A = np.log(np.maximum(self.A_, 1e-300))

        filtered_proba, predictive_ll = causal_forward_filter(log_pi, log_A, log_B)
        return filtered_proba, predictive_ll

    def smooth(self, X: np.ndarray) -> np.ndarray:
        """
        Compute retrospective smoothed state probabilities P(z_t | x_{1:T}).
        Explicitly labeled offline / retrospective.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fit before smoothing.")
        X = np.asarray(X, dtype=np.float64)
        log_B = compute_gaussian_log_emissions(
            X, self.means_, self.covs_, self.covariance_type, self.cov_regularization
        )
        log_pi = np.log(np.maximum(self.pi_, 1e-300))
        log_A = np.log(np.maximum(self.A_, 1e-300))

        log_alpha, total_ll = forward_log_recursion(log_pi, log_A, log_B)
        log_beta = backward_log_recursion(log_A, log_B)
        return compute_smoothed_posteriors(log_alpha, log_beta, total_ll)

    def decode_viterbi(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute retrospective MAP state sequence using Viterbi decoding.
        Explicitly labeled offline / retrospective.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fit before Viterbi decoding.")
        X = np.asarray(X, dtype=np.float64)
        log_B = compute_gaussian_log_emissions(
            X, self.means_, self.covs_, self.covariance_type, self.cov_regularization
        )
        log_pi = np.log(np.maximum(self.pi_, 1e-300))
        log_A = np.log(np.maximum(self.A_, 1e-300))

        return viterbi_decode(log_pi, log_A, log_B)

    def predict_next_proba(self, current_posterior: np.ndarray) -> np.ndarray:
        """
        Predict one-step ahead state distribution: P(z_{t+1}) = P(z_t) @ A.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fit before predicting.")
        return current_posterior @ self.A_

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model parameters and configuration to dictionary."""
        if not self.is_fitted_:
            raise RuntimeError("Cannot serialize unfitted model.")
        return {
            "schema_version": "1.0.0",
            "n_states": self.n_states,
            "covariance_type": self.covariance_type,
            "n_features": self.n_features_,
            "pi": self.pi_.tolist(),
            "A": self.A_.tolist(),
            "means": self.means_.tolist(),
            "covs": self.covs_.tolist(),
            "final_log_likelihood": self.final_log_likelihood_,
            "n_iter": self.n_iter_,
            "cov_regularization": self.cov_regularization,
            "min_effective_occupancy": self.min_effective_occupancy,
            "random_state": self.random_state,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GaussianHMM":
        """Reconstruct model from dictionary with parameter validation."""
        model = cls(
            n_states=d["n_states"],
            covariance_type=d["covariance_type"],
            cov_regularization=d.get("cov_regularization", 1e-4),
            min_effective_occupancy=d.get("min_effective_occupancy", 0.02),
            random_state=d.get("random_state", 42),
        )
        model.n_features_ = d["n_features"]
        model.pi_ = np.array(d["pi"], dtype=np.float64)
        model.A_ = np.array(d["A"], dtype=np.float64)
        model.means_ = np.array(d["means"], dtype=np.float64)
        model.covs_ = np.array(d["covs"], dtype=np.float64)
        model.final_log_likelihood_ = d.get("final_log_likelihood", float("-inf"))
        model.n_iter_ = d.get("n_iter", 0)

        # Validate parameters
        assert np.isclose(np.sum(model.pi_), 1.0, atol=1e-4)
        assert np.allclose(np.sum(model.A_, axis=1), 1.0, atol=1e-4)
        model.is_fitted_ = True
        return model
