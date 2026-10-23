"""
Financial metrics, bootstrap uncertainty, and event analysis.
"""

from regimeflow.metrics.bootstrap import paired_block_bootstrap
from regimeflow.metrics.events import analyze_event_windows
from regimeflow.metrics.financial import compute_financial_metrics

__all__ = [
    "compute_financial_metrics",
    "paired_block_bootstrap",
    "analyze_event_windows",
]
