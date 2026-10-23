"""
FastAPI service exposing causal forward filtering, regime inference, and visual dashboard.
Strictly enforces causal forward filtering over time-ordered history.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from regimeflow import __version__
from regimeflow.dashboard.template import get_dashboard_html
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


def _find_latest_run_dir(base_results_dir: str = "results") -> Optional[Path]:
    """Find the most recent experiment run directory containing outputs."""
    p = Path(base_results_dir)
    if not p.exists():
        return None
    runs = [d for d in p.iterdir() if d.is_dir() and (d / "strategy_metrics.csv").exists()]
    if not runs:
        return None
    return sorted(runs, key=lambda x: x.stat().st_mtime, reverse=True)[0]


def create_app(
    model: Optional[GaussianHMM] = None,
    run_dir: Optional[Path | str] = None,
) -> FastAPI:
    app = FastAPI(
        title="RegimeFlow Causal Regime API & Visual Dashboard",
        version=__version__,
        description="Probabilistic market regime detection and interactive visual dashboard.",
    )

    # In-memory model reference
    app.state.model = model
    
    # Locate experiment run directory
    resolved_run_dir = Path(run_dir) if run_dir else _find_latest_run_dir()
    app.state.run_dir = resolved_run_dir

    # Mount static figures directory if available
    if resolved_run_dir and (resolved_run_dir / "figures").exists():
        app.mount(
            "/static/figures",
            StaticFiles(directory=str(resolved_run_dir / "figures")),
            name="figures",
        )

    # 1. Visual Web Dashboard
    @app.get("/", response_class=HTMLResponse)
    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        return HTMLResponse(content=get_dashboard_html())

    # 2. System Health Check
    @app.get("/health")
    def health_check() -> Dict[str, Any]:
        return {
            "status": "healthy",
            "version": __version__,
            "model_loaded": app.state.model is not None,
            "n_states": app.state.model.n_states if app.state.model else None,
            "run_dir": str(app.state.run_dir) if app.state.run_dir else None,
        }

    # 3. Time Series API for interactive charts
    @app.get("/api/data/timeseries")
    def get_timeseries_data() -> Dict[str, Any]:
        rd = app.state.run_dir
        if rd is None or not (rd / "oos_predictions.csv").exists():
            raise HTTPException(status_code=404, detail="Experiment run data not found")

        oos_df = pd.read_csv(rd / "oos_predictions.csv", index_col=0)
        ret_df = pd.read_csv(rd / "strategy_returns.csv", index_col=0)

        # Merge for consistency and sanitize against NaN/inf
        merged = oos_df.join(ret_df, lsuffix="_oos", rsuffix="_ret", how="inner")
        merged = merged.ffill().bfill().fillna(0.0)

        # Downsample to ~500 points for smooth browser rendering
        step = max(1, len(merged) // 500)
        downsampled = merged.iloc[::step]

        # Calculate drawdowns
        def compute_dd(nav_series: pd.Series) -> List[float]:
            peaks = nav_series.cummax()
            dd = (nav_series - peaks) / np.maximum(peaks, 1e-8)
            dd = dd.fillna(0.0)
            return [float(x) for x in dd]

        prices_series = (
            downsampled["asset_price"]
            if "asset_price" in downsampled.columns
            else downsampled["nav_buy_and_hold"]
        )

        return {
            "dates": [str(d) for d in downsampled.index],
            "prices": [float(p) for p in prices_series],
            "high_risk_prob": [float(p) for p in downsampled["high_risk_prob"]],
            "nav_regime": [float(n) for n in downsampled["nav_regime_aware"]],
            "nav_bh": [float(n) for n in downsampled["nav_buy_and_hold"]],
            "nav_ma": [float(n) for n in downsampled["nav_moving_average"]],
            "dd_regime": compute_dd(downsampled["nav_regime_aware"]),
            "dd_bh": compute_dd(downsampled["nav_buy_and_hold"]),
            "total_points": len(downsampled),
        }

    # 4. Summary Metrics API
    @app.get("/api/data/summary")
    def get_summary_metrics() -> Dict[str, Any]:
        rd = app.state.run_dir
        if rd is None or not (rd / "strategy_metrics.csv").exists():
            raise HTTPException(status_code=404, detail="Strategy metrics not found")

        metrics_df = pd.read_csv(rd / "strategy_metrics.csv").set_index("strategy")
        metrics_df = metrics_df.fillna(0.0)
        return metrics_df.to_dict(orient="index")

    # 5. Event Windows API
    @app.get("/api/data/events")
    def get_event_metrics() -> List[Dict[str, Any]]:
        rd = app.state.run_dir
        if rd is None or not (rd / "event_window_metrics.csv").exists():
            return []

        events_df = pd.read_csv(rd / "event_window_metrics.csv")
        return events_df.to_dict(orient="records")

    # 6. Real-time Causal Regime Filter Inference
    @app.post("/predict-regime", response_model=FilterResponse)
    def predict_regime(req: FilterRequest) -> FilterResponse:
        active_model: Optional[GaussianHMM] = app.state.model
        if active_model is None or not active_model.is_fitted_:
            # Auto-fit a default 3-state HMM on synthetic benchmark if none loaded
            np_rng = np.random.default_rng(42)
            synth_x = np_rng.normal(size=(200, 3))
            fallback_model = GaussianHMM(n_states=3, covariance_type="full", n_restarts=1, max_iter=15)
            fallback_model.fit(synth_x)
            app.state.model = fallback_model
            active_model = fallback_model

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
