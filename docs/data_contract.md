# RegimeFlow Data Contract & Quality Specification

## 1. Primary Instrument & Data Sources
- **Ticker:** SPY (SPDR S&P 500 ETF Trust)
- **Target History:** 2000-01-01 through latest available completed trading session.
- **Primary Online Provider:** Yahoo Finance via `yfinance` (educational/research use).
- **Offline Synthetic Generator:** Deterministic multi-state financial time series generator simulating stochastic volatility, fat tails, and regime shifts.
- **Storage Policy:** Raw market data is saved strictly in local ignored directories (`data/raw/`, `data/cache/`). No proprietary or raw market data is committed to source control.

---

## 2. Input Data Schema

### 2.1 Raw OHLCV Schema
Each raw observation must conform to the following schema:

| Column | Type | Nullable | Description / Invariants |
|---|---|---|---|
| `date` | `pd.Timestamp` (UTC or naive ISO8601) | No | Monotonically strictly increasing, unique index |
| `open` | `float64` | No | Strict positivity ($> 0.0$) |
| `high` | `float64` | No | Strict positivity; $high \ge \max(open, close, low)$ |
| `low` | `float64` | No | Strict positivity; $low \le \min(open, close, high)$ |
| `close` | `float64` | No | Strict positivity |
| `adj_close` | `float64` | No | Strict positivity (corporate-action adjusted) |
| `volume` | `float64` | No | Non-negative ($\ge 0.0$) |

### 2.2 Price Adjustment Policy
To prevent inconsistent volatility estimates when computing range-based estimators (such as Garman-Klass or Parkinson):
- When corporate action adjustment is applied, an adjustment ratio is computed:
  $$k_t = \frac{\text{adj\_close}_t}{\text{close}_t}$$
- Adjusted OHLC prices are consistently defined as:
  $$\text{adj\_open}_t = k_t \cdot \text{open}_t, \quad \text{adj\_high}_t = k_t \cdot \text{high}_t, \quad \text{adj\_low}_t = k_t \cdot \text{low}_t, \quad \text{adj\_close}_t = k_t \cdot \text{close}_t$$
- This guarantees $adj\_low_t \le adj\_open_t, adj\_close_t \le adj\_high_t$ and prevents range corruption.

---

## 3. Data Quality Gates & Validation Rules

A dataset is accepted into the research pipeline if and only if it passes all quality gates:
1. **Monotonicity & Uniqueness:** Timestamps must be strictly ascending with zero duplicate timestamps.
2. **Positivity:** All price columns (`open`, `high`, `low`, `close`, `adj_close`) must be strictly positive ($> 0$).
3. **Bar Consistency:** For every session, $low_t \le \min(open_t, close_t)$ and $high_t \ge \max(open_t, close_t)$.
4. **Finite Values:** Zero occurrences of `NaN`, `inf`, or `-inf` in raw price fields.
5. **Session Gap Audit:** Gaps exceeding typical weekend/holiday gaps (> 4 calendar days) are audited and flagged in the data quality manifest.
6. **Minimum History:** Requires at least $876$ trading sessions (750 training sessions + 126 test sessions) for full walk-forward evaluation. Shorter synthetic datasets can be used for smoke tests with explicitly configured window sizes.

---

## 4. Cache & Provenance Manifest

Every cached dataset stored on disk is accompanied by a JSON sidecar manifest (`<dataset_name>.manifest.json`):
```json
{
  "manifest_version": "1.0.0",
  "dataset_id": "SPY_daily_2000_2026",
  "symbol": "SPY",
  "source": "yfinance",
  "retrieval_timestamp_utc": "2026-10-23T10:20:00Z",
  "start_date": "2000-01-03",
  "end_date": "2026-10-20",
  "row_count": 6745,
  "sha256_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "adjusted": true,
  "columns": ["date", "open", "high", "low", "close", "adj_close", "volume"]
}
```

If the cached file's calculated SHA-256 hash does not match the manifest, the cache is treated as corrupted and must be invalidated.

---

## 5. Synthetic Data Specification

For offline testing and algorithmic validation:
- **Known-State HMM Generator:** Generates discrete state sequence $z_t \sim \text{MarkovChain}(\pi, A)$ and continuous observations $x_t \sim \mathcal{N}(\mu_{z_t}, \Sigma_{z_t})$.
- **Synthetic Financial OHLCV Generator:** Uses a 3-regime Geometric Brownian Motion with stochastic drift, regime-dependent jump diffusion, and GARCH-like volatility clustering to create realistic synthetic candles with known regime changepoints.
- Both generators require an explicit integer `seed` to guarantee bitwise reproducibility.
