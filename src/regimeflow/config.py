"""
Configuration schemas and loaders for RegimeFlow.
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import yaml
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    symbol: str = "SPY"
    start_date: str = "2000-01-01"
    end_date: Optional[str] = None
    source: Literal["yfinance", "synthetic"] = "yfinance"
    data_dir: str = "data"
    cache_enabled: bool = True
    adjusted: bool = True
    synthetic_length: int = 1500
    seed: int = 42


class FeatureConfig(BaseModel):
    feature_group: Literal["base", "extended"] = "base"
    vol_short_window: int = 5
    vol_medium_window: int = 21
    vol_long_window: int = 63
    momentum_window: int = 21
    gk_window: int = 5
    volume_window: int = 63
    eps: float = 1e-8


class ModelConfig(BaseModel):
    n_states_candidates: List[int] = Field(default_factory=lambda: [2, 3, 4, 5])
    covariance_type: Literal["full", "diag"] = "full"
    n_restarts: int = 5
    max_iter: int = 100
    tol: float = 1e-4
    cov_regularization: float = 1e-4
    min_effective_occupancy: float = 0.02
    random_state: int = 42

    @field_validator("n_states_candidates")
    @classmethod
    def validate_states(cls, v: List[int]) -> List[int]:
        if not v or any(k < 2 for k in v):
            raise ValueError("All candidate states must be >= 2")
        return v


class WalkForwardConfig(BaseModel):
    train_window: int = 750
    test_window: int = 126
    step_size: int = 126
    scaler_type: Literal["standard", "robust"] = "standard"


class StrategyConfig(BaseModel):
    cash_return: float = 0.0
    transaction_cost_bps: float = 10.0  # 10 bps = 0.0010
    risk_threshold_multiplier: float = 1.0
    min_weight: float = 0.0
    max_weight: float = 1.0
    moving_average_window: int = 200


class EventWindowConfig(BaseModel):
    name: str
    start_date: str
    end_date: str
    description: str


class EvaluationConfig(BaseModel):
    bootstrap_block_length: int = 21
    bootstrap_replications: int = 1000
    bootstrap_seed: int = 42
    event_windows: List[EventWindowConfig] = Field(
        default_factory=lambda: [
            EventWindowConfig(
                name="GFC",
                start_date="2007-08-01",
                end_date="2009-06-30",
                description="Global Financial Crisis",
            ),
            EventWindowConfig(
                name="COVID",
                start_date="2020-02-19",
                end_date="2020-04-30",
                description="COVID-19 Market Shock",
            ),
            EventWindowConfig(
                name="INFLATION_2022",
                start_date="2022-01-03",
                end_date="2022-12-30",
                description="2022 Inflation & Rate Hike Drawdown",
            ),
        ]
    )


class ExperimentConfig(BaseModel):
    experiment_name: str = "regimeflow_primary"
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output_dir: str = "results"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
