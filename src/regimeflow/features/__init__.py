"""
Feature engineering, formulas, train-only scalers, and diagnostics.
"""

from regimeflow.features.diagnostics import compute_feature_diagnostics
from regimeflow.features.formulas import (
    compute_garman_klass_volatility,
    compute_log_returns,
    compute_parkinson_volatility,
    compute_realized_volatility,
    compute_trailing_momentum,
    compute_volatility_ratio,
    compute_volume_zscore,
)
from regimeflow.features.pipeline import FeaturePipeline
from regimeflow.features.schema import FEATURE_CATALOG, FeatureMeta, get_feature_names
from regimeflow.features.transforms import (
    TrainOnlyRobustScaler,
    TrainOnlyStandardScaler,
)

__all__ = [
    "compute_feature_diagnostics",
    "compute_log_returns",
    "compute_realized_volatility",
    "compute_volatility_ratio",
    "compute_trailing_momentum",
    "compute_garman_klass_volatility",
    "compute_parkinson_volatility",
    "compute_volume_zscore",
    "FeaturePipeline",
    "FEATURE_CATALOG",
    "FeatureMeta",
    "get_feature_names",
    "TrainOnlyStandardScaler",
    "TrainOnlyRobustScaler",
]
