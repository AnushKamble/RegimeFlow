"""
Data provider adapters, Yahoo Finance client, and the unified DataPipeline.
"""

from abc import ABC, abstractmethod
import time
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from regimeflow.config import DataConfig
from regimeflow.data.cache import CacheManager
from regimeflow.data.schema import DatasetManifest
from regimeflow.data.synthetic import generate_synthetic_market_data
from regimeflow.data.validation import DataQualityReport, validate_market_data
from regimeflow.logging_utils import logger


class BaseDataProvider(ABC):
    """Abstract base class for financial data providers."""

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """Fetch raw market data and return standardized DataFrame."""
        pass


class SyntheticProvider(BaseDataProvider):
    """Generates deterministic synthetic market data."""

    def __init__(self, n_sessions: int = 1500, seed: int = 42):
        self.n_sessions = n_sessions
        self.seed = seed

    def fetch(
        self,
        symbol: str = "SYNTHETIC",
        start_date: str = "2010-01-04",
        end_date: Optional[str] = None,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        logger.info(
            f"Generating synthetic market series ({self.n_sessions} sessions, seed={self.seed})..."
        )
        df, _ = generate_synthetic_market_data(
            n_sessions=self.n_sessions,
            start_date=start_date,
            seed=self.seed,
        )
        return df


class YFinanceProvider(BaseDataProvider):
    """Downloads historical daily data from Yahoo Finance via yfinance."""

    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0, timeout: int = 20):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

    def fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        import yfinance as yf

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    f"Fetching '{symbol}' from yfinance (attempt {attempt}/{self.max_retries}, "
                    f"start={start_date}, end={end_date})..."
                )
                raw = yf.download(
                    tickers=symbol,
                    start=start_date,
                    end=end_date,
                    auto_adjust=False,
                    progress=False,
                    timeout=self.timeout,
                )

                if raw is None or raw.empty:
                    raise ValueError(f"No data returned for '{symbol}' from yfinance.")

                # If multi-level columns (e.g. from yfinance 0.2+), flatten
                if isinstance(raw.columns, pd.MultiIndex):
                    # MultiIndex columns: Level 0 is Metric (Close, Open...), Level 1 is Ticker
                    if raw.columns.nlevels > 1:
                        raw.columns = raw.columns.get_level_values(0)

                raw.columns = [str(c).strip().lower().replace(" ", "_") for c in raw.columns]

                # Map column variations
                rename_map = {
                    "adj_close": "adj_close",
                    "adjclose": "adj_close",
                }
                raw = raw.rename(columns=rename_map)

                if "adj_close" not in raw.columns and "close" in raw.columns:
                    raw["adj_close"] = raw["close"]

                required = ["open", "high", "low", "close", "adj_close", "volume"]
                missing = [c for c in required if c not in raw.columns]
                if missing:
                    raise ValueError(f"Missing required columns from download: {missing}")

                df = raw[required].copy()
                df = df.dropna(how="all")

                # Remove timezone if present for clean daily operations
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)

                df.index = pd.to_datetime(df.index)
                df.index.name = "date"

                # Consistent corporate action adjustment
                if adjusted:
                    # k_t = adj_close / close
                    close_val = np.maximum(df["close"].to_numpy(dtype=np.float64), 1e-8)
                    k = df["adj_close"].to_numpy(dtype=np.float64) / close_val
                    df["open"] = df["open"] * k
                    df["high"] = df["high"] * k
                    df["low"] = df["low"] * k
                    df["close"] = df["adj_close"]

                return df

            except Exception as e:
                last_error = e
                logger.warning(f"Download attempt {attempt} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        raise RuntimeError(
            f"Failed to download '{symbol}' after {self.max_retries} attempts: {last_error}"
        )


class DataPipeline:
    """Coordinates fetching, caching, and data quality validation."""

    def __init__(self, cache_manager: Optional[CacheManager] = None):
        self.cache = cache_manager or CacheManager()

    def get_dataset_id(self, config: DataConfig) -> str:
        symbol = config.symbol.replace("^", "").replace("=", "_")
        start = config.start_date.replace("-", "")
        end = (config.end_date or "latest").replace("-", "")
        mode = config.source
        return f"{symbol}_{mode}_{start}_{end}"

    def load_or_fetch(
        self,
        config: DataConfig,
        force_refresh: bool = False,
    ) -> Tuple[pd.DataFrame, DatasetManifest, DataQualityReport]:
        """
        Load from cache if present; otherwise fetch, validate, and cache.
        """
        dataset_id = self.get_dataset_id(config)

        # 1. Check cache unless forced refresh
        if config.cache_enabled and not force_refresh:
            cached_df, cached_manifest = self.cache.load(dataset_id)
            if cached_df is not None and cached_manifest is not None:
                report = validate_market_data(cached_df)
                if report.is_valid:
                    logger.info(
                        f"Loaded validated dataset from cache '{dataset_id}' ({len(cached_df)} rows)."
                    )
                    return cached_df, cached_manifest, report
                else:
                    logger.warning(
                        f"Cached data for '{dataset_id}' failed quality gates: {report.errors}. Refetching..."
                    )

        # 2. Fetch from designated provider
        if config.source == "synthetic":
            provider = SyntheticProvider(
                n_sessions=config.synthetic_length,
                seed=config.seed,
            )
        elif config.source == "yfinance":
            provider = YFinanceProvider()
        else:
            raise ValueError(f"Unknown data source: {config.source}")

        df = provider.fetch(
            symbol=config.symbol,
            start_date=config.start_date,
            end_date=config.end_date,
            adjusted=config.adjusted,
        )

        # 3. Validate
        report = validate_market_data(df)
        if not report.is_valid:
            error_msg = f"Data validation failed for '{config.symbol}': {report.errors}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 4. Create manifest & cache
        manifest = DatasetManifest(
            symbol=config.symbol,
            source=config.source,
            start_date=report.start_date,
            end_date=report.end_date,
            row_count=len(df),
            adjusted=config.adjusted,
            metadata={"data_quality_report": report.to_dict()},
        )

        if config.cache_enabled:
            self.cache.save(dataset_id, df, manifest)

        return df, manifest, report
