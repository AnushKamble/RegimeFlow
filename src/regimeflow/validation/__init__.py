"""
Walk-forward evaluation, baselines, and leakage checks.
"""

from regimeflow.validation.baselines import StaticGaussianBaseline
from regimeflow.validation.folds import FoldBounds, generate_walk_forward_folds
from regimeflow.validation.leakage_checks import (
    assert_causal_invariance,
    assert_non_overlapping_folds,
)
from regimeflow.validation.walk_forward import WalkForwardEvaluator

__all__ = [
    "StaticGaussianBaseline",
    "FoldBounds",
    "generate_walk_forward_folds",
    "WalkForwardEvaluator",
    "assert_non_overlapping_folds",
    "assert_causal_invariance",
]
