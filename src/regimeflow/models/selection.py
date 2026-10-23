"""
Model selection engine evaluating candidate state counts via train-only BIC.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
import pandas as pd
from regimeflow.config import ModelConfig
from regimeflow.logging_utils import logger
from regimeflow.models.gaussian_hmm import GaussianHMM


@dataclass
class CandidateModelResult:
    n_states: int
    covariance_type: str
    n_parameters: int
    n_train_samples: int
    train_log_likelihood: float
    bic: float
    aic: float
    converged: bool
    iterations: int
    min_occupancy_fraction: float
    is_valid: bool
    rejection_reason: Optional[str] = None
    model: Optional[GaussianHMM] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("model", None)  # exclude model object from tabular export
        return d


class ModelSelector:
    """
    Evaluates candidate state counts K on training data using BIC.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.results_: List[CandidateModelResult] = []
        self.best_result_: Optional[CandidateModelResult] = None

    def fit_and_select(
        self,
        X_train: np.ndarray,
        candidates: Optional[List[int]] = None,
    ) -> Tuple[GaussianHMM, CandidateModelResult]:
        """
        Fit all candidate state counts on X_train and select model minimizing BIC.

        Parameters
        ----------
        X_train : np.ndarray of shape (N_train, d)
            Training feature observations.
        candidates : Optional[List[int]]
            Candidate state counts K (defaults to config.n_states_candidates).

        Returns
        -------
        best_model : GaussianHMM
            Selected model with lowest valid BIC.
        best_result : CandidateModelResult
            Metadata and diagnostic evaluation of selected candidate.
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        N_train, d = X_train.shape
        cand_list = candidates or self.config.n_states_candidates

        self.results_ = []
        best_bic = float("inf")
        selected_model: Optional[GaussianHMM] = None
        selected_res: Optional[CandidateModelResult] = None

        logger.info(
            f"Starting model selection over K in {cand_list} on {N_train} training samples "
            f"(covariance_type={self.config.covariance_type})..."
        )

        for K in cand_list:
            try:
                model = GaussianHMM(
                    n_states=K,
                    covariance_type=self.config.covariance_type,
                    max_iter=self.config.max_iter,
                    tol=self.config.tol,
                    n_restarts=self.config.n_restarts,
                    cov_regularization=self.config.cov_regularization,
                    min_effective_occupancy=self.config.min_effective_occupancy,
                    random_state=self.config.random_state,
                )
                model.fit(X_train)

                train_ll = model.final_log_likelihood_
                p = model.n_parameters
                bic = model.compute_bic(N_train, train_ll)
                aic = model.compute_aic(train_ll)

                # Check state occupancy on training set
                smoothed_post = model.smooth(X_train)
                occ_frac = float(np.min(np.sum(smoothed_post, axis=0)) / N_train)

                is_valid = True
                reason = None
                if occ_frac < self.config.min_effective_occupancy:
                    is_valid = False
                    reason = f"State occupancy {occ_frac:.4f} below threshold {self.config.min_effective_occupancy}"

                cand_res = CandidateModelResult(
                    n_states=K,
                    covariance_type=self.config.covariance_type,
                    n_parameters=p,
                    n_train_samples=N_train,
                    train_log_likelihood=train_ll,
                    bic=bic,
                    aic=aic,
                    converged=True,
                    iterations=model.n_iter_,
                    min_occupancy_fraction=occ_frac,
                    is_valid=is_valid,
                    rejection_reason=reason,
                    model=model,
                )

                if is_valid and bic < best_bic:
                    best_bic = bic
                    selected_model = model
                    selected_res = cand_res

            except Exception as e:
                logger.warning(f"Candidate K={K} failed: {e}")
                cand_res = CandidateModelResult(
                    n_states=K,
                    covariance_type=self.config.covariance_type,
                    n_parameters=0,
                    n_train_samples=N_train,
                    train_log_likelihood=float("-inf"),
                    bic=float("inf"),
                    aic=float("inf"),
                    converged=False,
                    iterations=0,
                    min_occupancy_fraction=0.0,
                    is_valid=False,
                    rejection_reason=str(e),
                    model=None,
                )

            self.results_.append(cand_res)

        if selected_model is None or selected_res is None:
            raise RuntimeError(
                f"No candidate model converged to a valid fit. Results: {[r.to_dict() for r in self.results_]}"
            )

        self.best_result_ = selected_res
        logger.info(
            f"Selected K={selected_res.n_states} as optimal under train-only BIC "
            f"(BIC={selected_res.bic:.2f}, LL={selected_res.train_log_likelihood:.2f}, p={selected_res.n_parameters})"
        )
        return selected_model, selected_res

    def get_summary_dataframe(self) -> pd.DataFrame:
        """Export candidate evaluation table as DataFrame."""
        records = [r.to_dict() for r in self.results_]
        return pd.DataFrame.from_records(records)
