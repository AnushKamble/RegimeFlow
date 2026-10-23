"""
Chronological walk-forward fold generation with strict temporal boundaries.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List
import pandas as pd


@dataclass
class FoldBounds:
    fold_idx: int
    train_start_idx: int
    train_end_idx: int  # inclusive
    test_start_idx: int
    test_end_idx: int  # inclusive
    train_start_date: str
    train_end_date: str
    test_start_date: str
    test_end_date: str
    n_train: int
    n_test: int

    def to_dict(self) -> Dict[str, str | int]:
        return asdict(self)


def generate_walk_forward_folds(
    dates: pd.DatetimeIndex,
    train_window: int = 750,
    test_window: int = 126,
    step_size: int = 126,
    expanding: bool = False,
) -> List[FoldBounds]:
    """
    Generate chronological, non-overlapping walk-forward test folds.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Sorted unique timestamps of the feature dataset.
    train_window : int
        Number of sessions in training window.
    test_window : int
        Number of sessions in test window.
    step_size : int
        Step size between successive test folds.
    expanding : bool
        If True, training window expands from t=0. If False, rolling window of fixed size.

    Returns
    -------
    folds : List[FoldBounds]
        List of fold boundary descriptors.
    """
    N = len(dates)
    folds = []
    fold_idx = 0

    curr_train_start = 0
    curr_train_end = train_window - 1

    while True:
        curr_test_start = curr_train_end + 1
        curr_test_end = curr_test_start + test_window - 1

        if curr_test_end >= N:
            # If the remaining test window is smaller than test_window, check if at least 21 days remain
            if N - curr_test_start >= 21:
                curr_test_end = N - 1
            else:
                break

        train_s = 0 if expanding else curr_train_start
        train_e = curr_train_end
        test_s = curr_test_start
        test_e = curr_test_end

        fb = FoldBounds(
            fold_idx=fold_idx,
            train_start_idx=train_s,
            train_end_idx=train_e,
            test_start_idx=test_s,
            test_end_idx=test_e,
            train_start_date=str(dates[train_s].date()),
            train_end_date=str(dates[train_e].date()),
            test_start_date=str(dates[test_s].date()),
            test_end_date=str(dates[test_e].date()),
            n_train=(train_e - train_s + 1),
            n_test=(test_e - test_s + 1),
        )
        folds.append(fb)

        # Advance window
        fold_idx += 1
        curr_train_start += step_size
        curr_train_end += step_size

    return folds
