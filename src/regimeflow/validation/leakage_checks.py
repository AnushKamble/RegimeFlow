"""
Automated causality and anti-leakage verification checks.
"""

import numpy as np
import pandas as pd
from regimeflow.validation.walk_forward import WalkForwardEvaluator


def assert_non_overlapping_folds(fold_manifest: pd.DataFrame) -> None:
    """
    Verify temporal separation and absence of test overlap in walk-forward folds.
    """
    for i in range(len(fold_manifest)):
        row = fold_manifest.iloc[i]
        train_end = pd.to_datetime(row["train_end_date"])
        test_start = pd.to_datetime(row["test_start_date"])
        assert train_end < test_start, (
            f"Fold {i}: train_end ({train_end}) >= test_start ({test_start})"
        )

        if i > 0:
            prev_row = fold_manifest.iloc[i - 1]
            prev_test_end = pd.to_datetime(prev_row["test_end_date"])
            assert test_start > prev_test_end, (
                f"Fold {i} overlaps with Fold {i-1}: test_start ({test_start}) <= prev_test_end ({prev_test_end})"
            )


def assert_causal_invariance(
    evaluator: WalkForwardEvaluator,
    features: pd.DataFrame,
    forward_returns: pd.Series,
    perturb_offset_from_end: int = 15,
) -> None:
    """
    CRITICAL CAUSALITY CHECK:
    Modifying the last N test sessions must NOT change ANY filtered probabilities
    or predictive likelihoods prior to that perturbation cutoff.
    """
    oos_orig, folds_orig, _ = evaluator.run(features, forward_returns)

    # Perturb the last N rows of features
    cutoff_date = features.index[-perturb_offset_from_end]
    features_perturbed = features.copy()
    features_perturbed.loc[features_perturbed.index >= cutoff_date] *= 5.0

    oos_perturbed, folds_perturbed, _ = evaluator.run(features_perturbed, forward_returns)

    # Check all predictions strictly before cutoff_date
    clean_dates = oos_orig.index[oos_orig.index < cutoff_date]
    assert len(clean_dates) > 0, "No dates available prior to cutoff."

    for col in ["high_risk_prob", "hmm_predictive_ll", "assigned_state"]:
        orig_vals = oos_orig.loc[clean_dates, col].to_numpy()
        pert_vals = oos_perturbed.loc[clean_dates, col].to_numpy()

        if col == "assigned_state":
            assert np.array_equal(orig_vals, pert_vals), (
                f"Assigned state changed before cutoff date {cutoff_date}!"
            )
        else:
            diff = np.abs(orig_vals - pert_vals)
            max_diff = np.max(diff)
            assert max_diff < 1e-10, (
                f"Leakage detected in column '{col}': max diff {max_diff} > 1e-10"
            )
