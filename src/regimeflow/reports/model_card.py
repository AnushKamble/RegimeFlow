"""
Programmatic export of model cards and technical specification manifests.
"""

from pathlib import Path
from typing import Any, Dict
from regimeflow import __version__
from regimeflow.logging_utils import logger


def generate_model_card_dict() -> Dict[str, Any]:
    return {
        "model_name": "RegimeFlow Gaussian Hidden Markov Model",
        "version": __version__,
        "author": "RegimeFlow Team",
        "type": "Multivariate Gaussian HMM with Log-Domain Inference",
        "intended_use": "Market regime detection and risk-control strategy analysis",
        "safeguards": [
            "Strict causal online forward filtering",
            "Zero look-ahead leakage in feature engineering",
            "Train-only preprocessing and model selection",
            "Signal-to-execution return lag (weight at t earns return at t+1)",
            "Turnover transaction costs deducted",
            "Non-IID circular moving-block bootstrap uncertainty",
        ],
        "disclaimer": "Academic and engineering research prototype only. No investment advice.",
    }


def save_model_card(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    card_dict = generate_model_card_dict()
    import json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(card_dict, f, indent=2)
    logger.info(f"Exported model card metadata to {output_path}")
    return output_path
