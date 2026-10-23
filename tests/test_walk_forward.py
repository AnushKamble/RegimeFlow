"""
Tests for chronological walk-forward evaluation, baselines, and causality invariance.
"""

import numpy as np
import pandas as pd
import pytest
from regimeflow.config import ExperimentConfig, FeatureConfig, ModelConfig, WalkForwardConfig
from regimeflow.data.synthetic import generate_synthetic_market_data
from regimeflow.features.pipeline import FeaturePipeline
from regimeflow.validation.folds import generate_walk_forward_folds
from regimeflow.validation.leakage_checks import (
    assert_causal_invariance,
    assert_non_overlapping_folds,
)
from regimeflow.validation.walk_forward import WalkForwardEvaluator


def test_walk_forward_fold_generation():
    dates = pd.bdate_range("2015-01-01", periods=1000)
    folds = generate_walk_forward_folds(
        dates=dates, train_window=500, test_window=100, step_size=100
    )

    assert len(folds) >= 4
    for i, f in enumerate(folds):
        assert f.train_end_idx < f.test_start_idx
        assert f.n_train == 500
        assert f.n_test <= 100
        if i > 0:
            assert f.test_start_idx == folds[i - 1].test_end_idx + 1


def test_walk_forward_execution_and_causal_invariance():
    # 1. Generate synthetic data
    df, _ = generate_synthetic_market_data(n_sessions=400, seed=42)
    pipe = FeaturePipeline(FeatureConfig(feature_group="base"))
    feat, fwd_ret = pipe.fit_transform(df)

    # 2. Configure small fast walk-forward test: train 180, test 60, 2 candidate states
    cfg = ExperimentConfig(
        model=ModelConfig(
            n_states_candidates=[2],
            covariance_type="full",
            n_restarts=1,
            max_iter=20,
            random_state=42,
        ),
        walk_forward=WalkForwardConfig(
            train_window=180,
            test_window=60,
            step_size=60,
        ),
    )

    evaluator = WalkForwardEvaluator(cfg)
    oos_preds, fold_manifest, fitted_artifacts = evaluator.run(feat, fwd_ret)

    # Verify fold manifest
    assert len(fold_manifest) >= 2
    assert_non_overlapping_folds(fold_manifest)
    assert "mean_oos_predictive_ll" in fold_manifest.columns
    assert "mean_baseline_predictive_ll" in fold_manifest.columns
    assert all(np.isfinite(fold_manifest["mean_oos_predictive_ll"]))

    # Verify OOS predictions
    assert len(oos_preds) > 0
    assert "high_risk_prob" in oos_preds.columns
    assert "hmm_predictive_ll" in oos_preds.columns
    assert "baseline_predictive_ll" in oos_preds.columns
    assert (oos_preds["high_risk_prob"] >= 0.0).all()
    assert (oos_preds["high_risk_prob"] <= 1.0).all()

    # 3. Assert Causality / Anti-Leakage
    assert_causal_invariance(evaluator, feat, fwd_ret, perturb_offset_from_end=15)
