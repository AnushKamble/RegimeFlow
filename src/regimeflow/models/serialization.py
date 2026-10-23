"""
Validated serialization and deserialization for Gaussian HMM model artifacts.
"""

import json
from pathlib import Path
from typing import Dict
from regimeflow.logging_utils import logger
from regimeflow.models.gaussian_hmm import GaussianHMM


def save_hmm_model(model: GaussianHMM, path: str | Path) -> Path:
    """Save fitted HMM model artifact to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.to_dict()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    logger.info(f"Saved GaussianHMM model to {path}")
    return path


def load_hmm_model(path: str | Path) -> GaussianHMM:
    """Load and validate HMM model artifact from JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        state = json.load(f)
    model = GaussianHMM.from_dict(state)
    logger.info(
        f"Loaded GaussianHMM model from {path} (K={model.n_states}, type={model.covariance_type})"
    )
    return model
