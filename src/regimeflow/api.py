"""
FastAPI service exposing causal forward filtering and regime inference.
Strictly enforces causal forward filtering over time-ordered history.
"""

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel, Field
from regimeflow import __version__
from regimeflow.models.gaussian_hmm import GaussianHMM


class ObservationBar(BaseModel):
    date: str
    features: List[float] = Field(..., description="Ordered feature values")


class FilterRequest(BaseModel):
    history: List[ObservationBar] = Field(
        ...,
        min_length=1,
        description="Time-ordered sequential feature history ending at current observation t",
    )
    risk_threshold_multiplier: float = 1.0


class FilterResponse(BaseModel):
    timestamp: str
    history_length: int
    assigned_state: int
    filtered_probabilities: List[float]
    high_risk_probability: float
    recommended_target_weight: float
    predictive_log_likelihood: float
    disclaimer: str = (
        "RESEARCH PROTOTYPE ONLY: This API provides statistical time-series estimates "
        "and does not constitute investment advice or live financial trading recommendations."
    )


def create_app(model: Optional[GaussianHMM] = None) -> FastAPI:
    app = FastAPI(
        title="RegimeFlow Causal Regime API",
        version=__version__,
        description="Probabilistic market regime detection and causal forward filtering engine.",
    )

    # In-memory model reference (or fallback default model)
    app.state.model = model

    @app.get("/health")
    def health_check() -> Dict[str, Any]:
        return {
            "status": "healthy",
            "version": __version__,
            "model_loaded": app.state.model is not None,
            "n_states": app.state.model.n_states if app.state.model else None,
        }

    @app.post("/predict-regime", response_model=FilterResponse)
    def predict_regime(req: FilterRequest) -> FilterResponse:
        active_model: Optional[GaussianHMM] = app.state.model
        if active_model is None or not active_model.is_fitted_:
            raise HTTPException(
                status_code=503,
                detail="Model is not loaded or fitted. Fit or load an HMM model before inference.",
            )

        # Extract features array in strict chronological order
        X = np.array([bar.features for bar in req.history], dtype=np.float64)
        if X.shape[1] != active_model.n_features_:
            raise HTTPException(
                status_code=400,
                detail=f"Feature dimension mismatch: expected {active_model.n_features_}, got {X.shape[1]}",
            )

        # Causal forward filter over sequential history
        filtered_probs, predictive_ll = active_model.filter(X)
        latest_probs = filtered_probs[-1]
        assigned_state = int(np.argmax(latest_probs))
        high_risk_p = float(latest_probs[-1])  # Canonical highest-risk state is K-1

        raw_w = 1.0 - req.risk_threshold_multiplier * high_risk_p
        target_weight = float(np.clip(raw_w, 0.0, 1.0))

        return FilterResponse(
            timestamp=req.history[-1].date,
            history_length=len(req.history),
            assigned_state=assigned_state,
            filtered_probabilities=[float(p) for p in latest_probs],
            high_risk_probability=high_risk_p,
            recommended_target_weight=target_weight,
            predictive_log_likelihood=float(predictive_ll[-1]),
        )

    return app


app = create_app()
