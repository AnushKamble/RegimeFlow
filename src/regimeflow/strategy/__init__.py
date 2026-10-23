"""
Strategy simulation, signals, and execution accounting.
"""

from regimeflow.strategy.backtest import BacktestEngine
from regimeflow.strategy.signals import (
    generate_buy_and_hold_signals,
    generate_moving_average_signals,
    generate_regime_signals,
)

__all__ = [
    "BacktestEngine",
    "generate_buy_and_hold_signals",
    "generate_moving_average_signals",
    "generate_regime_signals",
]
