"""
Statistical baselines for comparative evaluation.
"""

from typing import Tuple
import numpy as np
import scipy.linalg
from regimeflow.models.gaussian_emissions import regularize_covariance


class StaticGaussianBaseline:
    """
    Single-state static multivariate Gaussian density baseline.
    Evaluates whether multi-state regime modeling adds genuine predictive density
    beyond an unconditional Gaussian fit on the training window.
    """

    def __init__(self, cov_regularization: float = 1e-4):
        self.cov_regularization = cov_regularization
        self.mean_: np.ndarray | None = None
        self.cov_: np.ndarray | None = None
        self.L_: np.ndarray | None = None
        self.log_det_: float | None = None
        self.is_fitted_: bool = False

    def fit(self, X_train: np.ndarray) -> "StaticGaussianBaseline":
        X = np.asarray(X_train, dtype=np.float64)
        N, d = X.shape
        self.mean_ = np.mean(X, axis=0)
        cov_raw = np.cov(X, rowvar=False) if N > 1 else np.eye(d)
        if cov_raw.ndim == 0:
            cov_raw = np.array([[cov_raw]])

        cov_reg, L, log_det = regularize_covariance(
            cov_raw, reg_factor=self.cov_regularization
        )
        self.cov_ = cov_reg
        self.L_ = L
        self.log_det_ = log_det
        self.is_fitted_ = True
        return self

    def score_samples(self, X_test: np.ndarray) -> np.ndarray:
        """
        Evaluate predictive log-density log N(x_t | mean, cov) per test observation.
        """
        if not self.is_fitted_ or self.mean_ is None or self.L_ is None or self.log_det_ is None:
            raise RuntimeError("Baseline must be fit before scoring.")

        X = np.asarray(X_test, dtype=np.float64)
        T, d = X.shape
        const = -0.5 * d * np.log(2.0 * np.pi)

        diff = X - self.mean_
        Y = scipy.linalg.solve_triangular(self.L_, diff.T, lower=True)
        mahalanobis = np.sum(Y**2, axis=0)

        log_densities = const - 0.5 * self.log_det_ - 0.5 * mahalanobis
        return log_densities
