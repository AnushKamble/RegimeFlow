"""
Feature schema definitions and metadata registry.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Literal


@dataclass(frozen=True)
class FeatureMeta:
    name: str
    group: Literal["base", "extended"]
    formula: str
    unit: str
    lookback_sessions: int
    decision_time_availability: str  # e.g. "after close t"
    missing_data_rule: str

    def to_dict(self) -> Dict[str, str | int]:
        return asdict(self)


FEATURE_CATALOG: Dict[str, FeatureMeta] = {
    "log_return": FeatureMeta(
        name="log_return",
        group="base",
        formula="log(AdjClose_t / AdjClose_{t-1})",
        unit="daily log return",
        lookback_sessions=1,
        decision_time_availability="after close t",
        missing_data_rule="Drop warmup NaN at t=0",
    ),
    "realized_vol_21": FeatureMeta(
        name="realized_vol_21",
        group="base",
        formula="std(log_return_{t-20:t}) * sqrt(252)",
        unit="annualized volatility",
        lookback_sessions=21,
        decision_time_availability="after close t",
        missing_data_rule="Drop warmup sessions < 21",
    ),
    "vol_ratio": FeatureMeta(
        name="vol_ratio",
        group="base",
        formula="RV_5 / max(RV_63, eps)",
        unit="dimensionless ratio",
        lookback_sessions=63,
        decision_time_availability="after close t",
        missing_data_rule="Drop warmup sessions < 63",
    ),
    "momentum_21": FeatureMeta(
        name="momentum_21",
        group="extended",
        formula="(AdjClose_t / AdjClose_{t-21}) - 1",
        unit="trailing return fraction",
        lookback_sessions=21,
        decision_time_availability="after close t",
        missing_data_rule="Drop warmup sessions < 21",
    ),
    "garman_klass_5": FeatureMeta(
        name="garman_klass_5",
        group="extended",
        formula="sqrt(mean(0.5*(log(H/L))^2 - (2*log(2)-1)*(log(C/O))^2) * 252)",
        unit="annualized range volatility",
        lookback_sessions=5,
        decision_time_availability="after close t",
        missing_data_rule="Drop warmup sessions < 5; floor negative variance at eps",
    ),
    "volume_zscore_63": FeatureMeta(
        name="volume_zscore_63",
        group="extended",
        formula="(log(1+Vol_t) - mean_63) / max(std_63, eps)",
        unit="z-score",
        lookback_sessions=63,
        decision_time_availability="after close t",
        missing_data_rule="Drop warmup sessions < 63",
    ),
}


def get_feature_names(group: Literal["base", "extended"]) -> List[str]:
    if group == "base":
        return [k for k, v in FEATURE_CATALOG.items() if v.group == "base"]
    return list(FEATURE_CATALOG.keys())
