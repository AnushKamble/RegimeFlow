"""
Tests for FastAPI causal regime filtering service.
"""

from fastapi.testclient import TestClient
import numpy as np
import pytest
from regimeflow.api import create_app
from regimeflow.models.gaussian_hmm import GaussianHMM


def test_api_health_endpoint():
    app = create_app()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is False


def test_api_predict_regime_flow():
    # 1. Fit a small 2-state HMM
    X = np.random.default_rng(42).normal(size=(50, 2))
    model = GaussianHMM(n_states=2, covariance_type="full", n_restarts=1, max_iter=10)
    model.fit(X)

    app = create_app(model=model)
    client = TestClient(app)

    # 2. Sequential observation request
    payload = {
        "history": [
            {"date": "2023-01-03", "features": [0.01, 0.15]},
            {"date": "2023-01-04", "features": [-0.02, 0.25]},
            {"date": "2023-01-05", "features": [0.005, 0.12]},
        ],
        "risk_threshold_multiplier": 1.0,
    }

    response = client.post("/predict-regime", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["timestamp"] == "2023-01-05"
    assert data["history_length"] == 3
    assert data["assigned_state"] in [0, 1]
    assert len(data["filtered_probabilities"]) == 2
    assert np.isclose(sum(data["filtered_probabilities"]), 1.0)
    assert 0.0 <= data["high_risk_probability"] <= 1.0
    assert 0.0 <= data["recommended_target_weight"] <= 1.0
    assert "RESEARCH PROTOTYPE" in data["disclaimer"]


def test_api_dimension_mismatch():
    X = np.random.default_rng(42).normal(size=(50, 3))
    model = GaussianHMM(n_states=2, covariance_type="full", n_restarts=1, max_iter=10)
    model.fit(X)

    app = create_app(model=model)
    client = TestClient(app)

    # Payload with 2 features when model expects 3
    payload = {
        "history": [
            {"date": "2023-01-03", "features": [0.01, 0.15]},
        ],
    }
    response = client.post("/predict-regime", json=payload)
    assert response.status_code == 400
    assert "dimension mismatch" in response.json()["detail"].lower()
