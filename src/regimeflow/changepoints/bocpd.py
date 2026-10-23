"""
Bayesian Online Changepoint Detection (Adams & MacKay 2007).
Exact online recursive run-length posterior evaluation with conjugate Student-t predictive density.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from scipy.special import gammaln, logsumexp


class BOCPD:
    """
    Bayesian Online Changepoint Detector for univariate Gaussian observations.

    Parameters
    ----------
    hazard_lambda : float
        Timescale parameter for geometric hazard function: H(r) = 1 / hazard_lambda.
        Default 200 sessions.
    mu0 : float
        Prior mean.
    kappa0 : float
        Prior sample size for mean.
    alpha0 : float
        Prior shape parameter for gamma precision.
    beta0 : float
        Prior rate parameter for gamma precision.
    max_run_length : int
        Maximum run-length to track before truncation (for efficiency).
    """

    def __init__(
        self,
        hazard_lambda: float = 200.0,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        max_run_length: int = 500,
    ):
        self.hazard_lambda = hazard_lambda
        self.H = 1.0 / hazard_lambda
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.max_run_length = max_run_length

    def _student_t_log_pdf(
        self,
        x: float,
        mu: np.ndarray,
        kappa: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate log predictive density of Student-t distribution under conjugate Normal-Gamma:
        degrees of freedom nu = 2 * alpha
        scale sigma2 = beta * (kappa + 1) / (alpha * kappa)
        """
        nu = 2.0 * alpha
        scale_sq = (beta * (kappa + 1.0)) / (alpha * kappa)
        scale_sq = np.maximum(scale_sq, 1e-12)

        const = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(np.pi * nu * scale_sq)
        log_diff = -0.5 * (nu + 1.0) * np.log1p(((x - mu) ** 2) / (nu * scale_sq))
        return const + log_diff

    def run(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run online recursive inference through the sequence.

        Parameters
        ----------
        observations : np.ndarray of shape (T,)
            Univariate time-series observations.

        Returns
        -------
        R : np.ndarray of shape (T, max_run_length + 1)
            Run-length posterior matrix R[t, r] = P(r_t = r | x_{1:t}).
        cp_probs : np.ndarray of shape (T,)
            Probability of a changepoint occurring at time t: P(r_t = 0 | x_{1:t}).
        """
        X = np.asarray(observations, dtype=np.float64).flatten()
        T = len(X)
        max_r = min(self.max_run_length, T)

        # Log joint distribution: log P(r_t = r, x_{1:t})
        log_R = np.full((T, max_r + 1), -np.inf, dtype=np.float64)
        cp_probs = np.zeros(T, dtype=np.float64)

        # Sufficient statistics vectors for each run length
        mu = np.array([self.mu0], dtype=np.float64)
        kappa = np.array([self.kappa0], dtype=np.float64)
        alpha = np.array([self.alpha0], dtype=np.float64)
        beta = np.array([self.beta0], dtype=np.float64)

        # t = 0: P(r_0 = 0) = 1.0
        log_R[0, 0] = 0.0
        cp_probs[0] = 1.0

        for t in range(1, T):
            x = X[t]
            curr_max_r = min(t, max_r)
            active_r = np.arange(curr_max_r)

            # 1. Evaluate predictive probabilities under each active run length
            log_pred = self._student_t_log_pdf(
                x, mu[:curr_max_r], kappa[:curr_max_r], alpha[:curr_max_r], beta[:curr_max_r]
            )

            # 2. Previous log probabilities
            log_prev = log_R[t - 1, :curr_max_r]

            # 3. Growth probability: r_t = r + 1
            log_growth = log_prev + log_pred + np.log(1.0 - self.H)

            # 4. Changepoint probability: r_t = 0
            log_cp = logsumexp(log_prev + log_pred + np.log(self.H))

            # Store in log_R
            log_R[t, 0] = log_cp
            log_R[t, 1 : curr_max_r + 1] = log_growth

            # 5. Normalize log_R[t] to obtain posterior P(r_t = r | x_{1:t})
            log_evidence = logsumexp(log_R[t, : curr_max_r + 1])
            log_R[t, : curr_max_r + 1] -= log_evidence

            # Normalized P(r_t = 0 | x_{1:t})
            cp_probs[t] = float(np.exp(log_R[t, 0]))

            # 6. Update hyper-parameters
            new_mu = np.zeros(curr_max_r + 1, dtype=np.float64)
            new_kappa = np.zeros(curr_max_r + 1, dtype=np.float64)
            new_alpha = np.zeros(curr_max_r + 1, dtype=np.float64)
            new_beta = np.zeros(curr_max_r + 1, dtype=np.float64)

            # Prior for r = 0
            new_mu[0] = self.mu0
            new_kappa[0] = self.kappa0
            new_alpha[0] = self.alpha0
            new_beta[0] = self.beta0

            # Updates for r >= 1:
            k_prev = kappa[:curr_max_r]
            m_prev = mu[:curr_max_r]
            a_prev = alpha[:curr_max_r]
            b_prev = beta[:curr_max_r]

            new_kappa[1:] = k_prev + 1.0
            new_mu[1:] = (k_prev * m_prev + x) / (k_prev + 1.0)
            new_alpha[1:] = a_prev + 0.5
            new_beta[1:] = b_prev + (k_prev * (x - m_prev) ** 2) / (2.0 * (k_prev + 1.0))

            mu = new_mu
            kappa = new_kappa
            alpha = new_alpha
            beta = new_beta

        R = np.exp(log_R)
        return R, cp_probs
