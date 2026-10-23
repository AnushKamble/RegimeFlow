"""
Feature engineering pipeline assembling validated market data into feature sets.
"""

from typing import Tuple
import pandas as pd
from regimeflow.config import FeatureConfig
from regimeflow.features.formulas import (
    compute_garman_klass_volatility,
    compute_log_returns,
    compute_realized_volatility,
    compute_trailing_momentum,
    compute_volatility_ratio,
    compute_volume_zscore,
)
from regimeflow.features.schema import get_feature_names
from regimeflow.logging_utils import logger


class FeaturePipeline:
    """
    Transforms market OHLCV data into backward-looking financial features.
    """

    def __init__(self, config: FeatureConfig):
        self.config = config

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build features and aligned 1-step forward asset returns.

        Parameters
        ----------
        df : pd.DataFrame
            Market data with columns ['open', 'high', 'low', 'close', 'adj_close', 'volume'].

        Returns
        -------
        features : pd.DataFrame
            Clean feature DataFrame with warmup periods removed.
        forward_returns : pd.Series
            1-step forward log return: r_{t+1} = log(adj_close_{t+1} / adj_close_t),
            indexed at t (the decision timestamp).
        """
        prices = df["adj_close"]
        log_ret = compute_log_returns(prices)

        # 1. Base features
        rv_short = compute_realized_volatility(
            log_ret, window=self.config.vol_short_window, annualize=True
        )
        rv_med = compute_realized_volatility(
            log_ret, window=self.config.vol_medium_window, annualize=True
        )
        rv_long = compute_realized_volatility(
            log_ret, window=self.config.vol_long_window, annualize=True
        )
        vol_ratio = compute_volatility_ratio(rv_short, rv_long, eps=self.config.eps)

        feature_dict = {
            "log_return": log_ret,
            "realized_vol_21": rv_med,
            "vol_ratio": vol_ratio,
        }

        # 2. Extended features
        if self.config.feature_group == "extended":
            mom = compute_trailing_momentum(prices, window=self.config.momentum_window)
            gk_vol = compute_garman_klass_volatility(
                open_prices=df["open"],
                high_prices=df["high"],
                low_prices=df["low"],
                close_prices=df["close"],
                window=self.config.gk_window,
                eps=self.config.eps,
            )
            vol_z = compute_volume_zscore(
                volume=df["volume"],
                window=self.config.volume_window,
                eps=self.config.eps,
            )
            feature_dict["momentum_21"] = mom
            feature_dict["garman_klass_5"] = gk_vol
            feature_dict["volume_zscore_63"] = vol_z

        features_raw = pd.DataFrame(feature_dict, index=df.index)

        # 3. Calculate 1-step forward return for backtesting evaluation
        # r_{t+1} = log(P_{t+1} / P_t) aligned at index t
        forward_returns = log_ret.shift(-1)

        # Combine to find common non-null rows (warmup period removal)
        # Note: we drop rows where features are NaN. The very last row has forward_returns=NaN,
        # which is expected for live/evaluation at the newest timestamp.
        feature_cols = list(features_raw.columns)
        valid_feature_mask = features_raw.notna().all(axis=1)

        features = features_raw.loc[valid_feature_mask].copy()
        forward_returns = forward_returns.loc[features.index]

        logger.info(
            f"Constructed feature set '{self.config.feature_group}' with {len(feature_cols)} features: "
            f"{feature_cols}. Usable sessions after warmup: {len(features)} / {len(df)}."
        )
        return features, forward_returns
