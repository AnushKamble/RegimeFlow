"""
Feature diagnostics: correlation, conditioning, heavy tails, and summary metrics.
"""

from typing import Any, Dict
import numpy as np
import pandas as pd


def compute_feature_diagnostics(features: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute distribution metrics, correlations, and condition number of the feature space.
    """
    if features.empty:
        return {}

    corr_matrix = features.corr()
    cov_matrix = np.cov(features.to_numpy(), rowvar=False)

    # Condition number of covariance matrix
    try:
        eigenvals = np.linalg.eigvalsh(cov_matrix)
        eigenvals = np.maximum(eigenvals, 1e-12)
        condition_number = float(np.max(eigenvals) / np.min(eigenvals))
    except Exception:
        condition_number = float("nan")

    desc = features.describe().to_dict()

    return {
        "n_samples": len(features),
        "n_features": features.shape[1],
        "feature_names": list(features.columns),
        "condition_number": condition_number,
        "correlation_matrix": corr_matrix.to_dict(),
        "summary_statistics": desc,
    }
