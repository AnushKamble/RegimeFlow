"""
Walk-forward chronological evaluation harness with strict temporal boundaries and no look-ahead.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from regimeflow.config import ExperimentConfig
from regimeflow.features.transforms import (
    TrainOnlyRobustScaler,
    TrainOnlyStandardScaler,
)
from regimeflow.logging_utils import logger
from regimeflow.models.gaussian_hmm import GaussianHMM
from regimeflow.models.selection import ModelSelector
from regimeflow.states.alignment import (
    compute_canonical_risk_permutation,
    permute_model_parameters,
)
from regimeflow.validation.baselines import StaticGaussianBaseline
from regimeflow.validation.folds import FoldBounds, generate_walk_forward_folds


class WalkForwardEvaluator:
    """
    Executes chronological out-of-sample walk-forward validation.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run(
        self,
        features: pd.DataFrame,
        forward_returns: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
        """
        Run complete chronological walk-forward evaluation.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix indexed by datetime.
        forward_returns : pd.Series
            1-step forward asset returns aligned with features.

        Returns
        -------
        oos_predictions : pd.DataFrame
            Out-of-sample timestamped causal filtered probabilities and predictive metrics.
        fold_manifest : pd.DataFrame
            Fold boundary descriptors and per-fold performance metrics.
        fitted_models : List[Dict[str, Any]]
            Fold-specific models and scalers.
        """
        dates = features.index
        wf_cfg = self.config.walk_forward

        folds = generate_walk_forward_folds(
            dates=dates,
            train_window=wf_cfg.train_window,
            test_window=wf_cfg.test_window,
            step_size=wf_cfg.step_size,
        )

        if not folds:
            raise ValueError(
                f"Dataset length {len(features)} is insufficient for walk-forward settings: "
                f"train_window={wf_cfg.train_window}, test_window={wf_cfg.test_window}."
            )

        logger.info(f"Generated {len(folds)} chronological walk-forward folds.")

        fold_records: List[Dict[str, Any]] = []
        oos_rows: List[Dict[str, Any]] = []
        fitted_artifacts: List[Dict[str, Any]] = []

        for fold in folds:
            logger.info(
                f"Evaluating Fold {fold.fold_idx + 1}/{len(folds)}: "
                f"Train [{fold.train_start_date} -> {fold.train_end_date}] ({fold.n_train} days), "
                f"Test [{fold.test_start_date} -> {fold.test_end_date}] ({fold.n_test} days)..."
            )

            # 1. Slice data
            X_train_df = features.iloc[fold.train_start_idx : fold.train_end_idx + 1]
            X_test_df = features.iloc[fold.test_start_idx : fold.test_end_idx + 1]

            # 2. Train-only preprocessing
            if wf_cfg.scaler_type == "robust":
                scaler = TrainOnlyRobustScaler()
            else:
                scaler = TrainOnlyStandardScaler()

            X_train = scaler.fit_transform(X_train_df)
            X_test = scaler.transform(X_test_df)

            # 3. Fit baseline: static single-state Gaussian on training data
            baseline = StaticGaussianBaseline(cov_regularization=self.config.model.cov_regularization)
            baseline.fit(X_train)
            baseline_test_ll = baseline.score_samples(X_test)

            # 4. Fit candidate HMMs on X_train only and select best K via BIC
            selector = ModelSelector(self.config.model)
            model, best_res = selector.fit_and_select(X_train)

            # 5. Canonical alignment: sort states by volatility (State 0 = calmest, State K-1 = highest risk)
            perm = compute_canonical_risk_permutation(
                model.means_, model.covs_, model.covariance_type, vol_feature_idx=0
            )
            pi_p, A_p, means_p, covs_p = permute_model_parameters(
                model.pi_, model.A_, model.means_, model.covs_, perm
            )
            model.pi_, model.A_, model.means_, model.covs_ = pi_p, A_p, means_p, covs_p

            # 6. Initialize filter from training data (no future test data)
            train_filtered, _ = model.filter(X_train)
            init_test_prior = train_filtered[-1] @ model.A_

            # 7. Causal forward filtering on test slice
            test_filtered, test_pred_ll = model.filter(X_test, initial_prior=init_test_prior)

            mean_hmm_pred_ll = float(np.mean(test_pred_ll))
            mean_base_pred_ll = float(np.mean(baseline_test_ll))

            fold_info = {
                **fold.to_dict(),
                "selected_k": best_res.n_states,
                "covariance_type": best_res.covariance_type,
                "n_parameters": best_res.n_parameters,
                "train_log_likelihood": best_res.train_log_likelihood,
                "bic": best_res.bic,
                "aic": best_res.aic,
                "mean_oos_predictive_ll": mean_hmm_pred_ll,
                "mean_baseline_predictive_ll": mean_base_pred_ll,
                "ll_improvement_over_baseline": mean_hmm_pred_ll - mean_base_pred_ll,
            }
            fold_records.append(fold_info)

            # 8. Record observation-level OOS predictions
            test_dates = X_test_df.index
            for t_idx, dt in enumerate(test_dates):
                filt_probs = test_filtered[t_idx]
                assigned_state = int(np.argmax(filt_probs))
                # High risk state is the highest volatility canonical state (state K-1)
                high_risk_prob = float(filt_probs[-1])
                fwd_ret = (
                    float(forward_returns.loc[dt])
                    if dt in forward_returns.index and not np.isnan(forward_returns.loc[dt])
                    else float("nan")
                )

                row_dict = {
                    "date": dt,
                    "fold_idx": fold.fold_idx,
                    "selected_k": best_res.n_states,
                    "assigned_state": assigned_state,
                    "high_risk_prob": high_risk_prob,
                    "hmm_predictive_ll": float(test_pred_ll[t_idx]),
                    "baseline_predictive_ll": float(baseline_test_ll[t_idx]),
                    "forward_asset_return": fwd_ret,
                }
                # Store individual state probabilities
                for k in range(best_res.n_states):
                    row_dict[f"prob_state_{k}"] = float(filt_probs[k])

                oos_rows.append(row_dict)

            fitted_artifacts.append({
                "fold_idx": fold.fold_idx,
                "model": model,
                "scaler": scaler,
                "selector_summary": selector.get_summary_dataframe(),
            })

        oos_predictions_df = pd.DataFrame(oos_rows).set_index("date")
        fold_manifest_df = pd.DataFrame(fold_records)

        logger.info(
            f"Completed walk-forward evaluation: {len(oos_predictions_df)} OOS sessions across "
            f"{len(fold_manifest_df)} folds. "
            f"Average OOS Predictive LL: HMM={fold_manifest_df['mean_oos_predictive_ll'].mean():.4f}, "
            f"Baseline={fold_manifest_df['mean_baseline_predictive_ll'].mean():.4f}."
        )

        return oos_predictions_df, fold_manifest_df, fitted_artifacts
