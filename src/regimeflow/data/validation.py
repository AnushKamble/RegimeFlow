"""
Data validation and quality assurance for RegimeFlow.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from regimeflow.data.schema import REQUIRED_COLUMNS


@dataclass
class DataQualityReport:
    is_valid: bool = True
    row_count: int = 0
    start_date: str = ""
    end_date: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    anomalous_gaps: List[Dict[str, Any]] = field(default_factory=list)
    summary_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_market_data(
    df: pd.DataFrame,
    min_sessions: int = 100,
    max_allowed_calendar_gap_days: int = 5,
) -> DataQualityReport:
    """
    Validate OHLCV market data against strict quality gates.

    Parameters
    ----------
    df : pd.DataFrame
        Input market data.
    min_sessions : int
        Minimum required trading sessions.
    max_allowed_calendar_gap_days : int
        Threshold beyond normal weekend/holiday to flag potential data gaps.

    Returns
    -------
    DataQualityReport
        Validation results and diagnostic metadata.
    """
    report = DataQualityReport()

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        report.is_valid = False
        report.errors.append("Dataset is empty or not a valid DataFrame.")
        return report

    report.row_count = len(df)

    # 1. Check required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        report.is_valid = False
        report.errors.append(f"Missing required columns: {missing_cols}")
        return report

    # 2. Check Index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            report.is_valid = False
            report.errors.append(f"Index cannot be parsed as DatetimeIndex: {e}")
            return report

    report.start_date = str(df.index.min().date())
    report.end_date = str(df.index.max().date())

    # 3. Check monotonicity and duplicates
    if not df.index.is_monotonic_increasing:
        report.is_valid = False
        report.errors.append("Dates are not monotonically strictly increasing.")

    if df.index.has_duplicates:
        dup_count = int(df.index.duplicated().sum())
        report.is_valid = False
        report.errors.append(f"Found {dup_count} duplicate timestamps in index.")

    # 4. Check minimum session count
    if len(df) < min_sessions:
        report.is_valid = False
        report.errors.append(
            f"History length {len(df)} is below minimum required {min_sessions} sessions."
        )

    # 5. Check finite values
    for col in REQUIRED_COLUMNS:
        non_finite = (~np.isfinite(df[col].to_numpy())).sum()
        if non_finite > 0:
            report.is_valid = False
            report.errors.append(f"Column '{col}' contains {non_finite} non-finite/NaN values.")

    # 6. Check positivity
    price_cols = ["open", "high", "low", "close", "adj_close"]
    for col in price_cols:
        min_val = float(df[col].min())
        if min_val <= 0:
            report.is_valid = False
            report.errors.append(f"Column '{col}' contains non-positive values (min={min_val}).")

    if (df["volume"] < 0).any():
        report.is_valid = False
        report.errors.append("Column 'volume' contains negative values.")

    # 7. Check bar consistency
    # Low <= min(Open, Close) and High >= max(Open, Close)
    tol = 1e-6
    bar_min = np.minimum(df["open"].to_numpy(), df["close"].to_numpy())
    bar_max = np.maximum(df["open"].to_numpy(), df["close"].to_numpy())
    low_violations = (df["low"].to_numpy() > bar_min + tol).sum()
    high_violations = (df["high"].to_numpy() < bar_max - tol).sum()

    if low_violations > 0:
        report.is_valid = False
        report.errors.append(f"Found {low_violations} sessions where Low > min(Open, Close).")

    if high_violations > 0:
        report.is_valid = False
        report.errors.append(f"Found {high_violations} sessions where High < max(Open, Close).")

    # 8. Check calendar gaps
    if len(df.index) > 1:
        time_diffs = (df.index[1:] - df.index[:-1]).days
        gap_indices = np.where(time_diffs > max_allowed_calendar_gap_days)[0]
        for idx in gap_indices:
            gap_info = {
                "start": str(df.index[idx].date()),
                "end": str(df.index[idx + 1].date()),
                "gap_days": int(time_diffs[idx]),
            }
            report.anomalous_gaps.append(gap_info)

        if report.anomalous_gaps:
            report.warnings.append(
                f"Found {len(report.anomalous_gaps)} anomalous date gaps > {max_allowed_calendar_gap_days} calendar days."
            )

    # 9. Compute summary statistics
    report.summary_stats = {
        "mean_close": float(df["close"].mean()),
        "min_close": float(df["close"].min()),
        "max_close": float(df["close"].max()),
        "mean_volume": float(df["volume"].mean()),
        "total_sessions": len(df),
    }

    return report
