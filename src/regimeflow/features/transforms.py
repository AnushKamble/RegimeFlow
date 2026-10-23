"""
Train-only feature scalers with explicit parameter state preservation.
Ensures zero data leakage between training and testing folds.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class BaseTrainOnlyScaler(ABC):
    """Base class for train-only feature transformers."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False

    @abstractmethod
    def fit(self, X: pd.DataFrame | np.ndarray) -> "BaseTrainOnlyScaler":
        """Fit transformer parameters on training data."""

    @abstractmethod
    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Apply fitted transformation to observations."""

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class TrainOnlyStandardScaler(BaseTrainOnlyScaler):
    """
    Standardizes features by removing the mean and scaling to unit variance.
    Fit parameters (mean, std) are derived strictly from training observations.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__(eps=eps)
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame | np.ndarray) -> "TrainOnlyStandardScaler":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            arr = X.to_numpy(dtype=np.float64)
        else:
            arr = np.asarray(X, dtype=np.float64)

        if len(arr) == 0:
            raise ValueError("Cannot fit scaler on empty array.")

        self.mean_ = np.mean(arr, axis=0)
        std_raw = np.std(arr, axis=0, ddof=1) if len(arr) > 1 else np.ones(arr.shape[1])
        self.std_ = np.maximum(std_raw, self.eps)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted_ or self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler must be fit before transforming.")

        if isinstance(X, pd.DataFrame):
            arr = X.to_numpy(dtype=np.float64)
        else:
            arr = np.asarray(X, dtype=np.float64)

        return (arr - self.mean_) / self.std_

    def to_dict(self) -> Dict[str, list]:
        if not self.is_fitted_:
            return {}
        return {
            "type": "standard",
            "feature_names": self.feature_names_,
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "TrainOnlyStandardScaler":
        scaler = cls(eps=d.get("eps", 1e-8))
        scaler.feature_names_ = d.get("feature_names", [])
        scaler.mean_ = np.array(d["mean"], dtype=np.float64)
        scaler.std_ = np.array(d["std"], dtype=np.float64)
        scaler.is_fitted_ = True
        return scaler


class TrainOnlyRobustScaler(BaseTrainOnlyScaler):
    """
    Scales features using median and interquartile range (IQR = Q3 - Q1).
    Robust to extreme outliers in heavy-tailed financial returns.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__(eps=eps)
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame | np.ndarray) -> "TrainOnlyRobustScaler":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            arr = X.to_numpy(dtype=np.float64)
        else:
            arr = np.asarray(X, dtype=np.float64)

        if len(arr) == 0:
            raise ValueError("Cannot fit scaler on empty array.")

        self.center_ = np.median(arr, axis=0)
        q25 = np.percentile(arr, 25, axis=0)
        q75 = np.percentile(arr, 75, axis=0)
        iqr = q75 - q25
        self.scale_ = np.maximum(iqr, self.eps)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted_ or self.center_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fit before transforming.")

        if isinstance(X, pd.DataFrame):
            arr = X.to_numpy(dtype=np.float64)
        else:
            arr = np.asarray(X, dtype=np.float64)

        return (arr - self.center_) / self.scale_

    def to_dict(self) -> Dict[str, list]:
        if not self.is_fitted_:
            return {}
        return {
            "type": "robust",
            "feature_names": self.feature_names_,
            "center": self.center_.tolist(),
            "scale": self.scale_.tolist(),
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "TrainOnlyRobustScaler":
        scaler = cls(eps=d.get("eps", 1e-8))
        scaler.feature_names_ = d.get("feature_names", [])
        scaler.center_ = np.array(d["center"], dtype=np.float64)
        scaler.scale_ = np.array(d["scale"], dtype=np.float64)
        scaler.is_fitted_ = True
        return scaler
