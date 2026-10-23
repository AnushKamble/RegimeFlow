# RegimeFlow System Architecture

## 1. Overview
RegimeFlow is a probabilistic market-regime detection and changepoint engine built for statistical ML and quantitative research. Its core contribution is an uncompromising separation of **causal/online inference** from **retrospective/offline smoothing**, paired with a strictly chronological walk-forward evaluation protocol.

```mermaid
flowchart TD
    subgraph Ingestion["1. Ingestion & Quality Gates"]
        P[Market Data Provider / Synthetic Generator] --> C[Disk Cache + SHA-256 Manifest]
        C --> V[Data Validation & QA Report]
    end

    subgraph FeatureEng["2. Feature Engineering"]
        V --> F[Backward-Looking Feature Builder]
        F --> FS[Feature Schema & Quality Artifacts]
    end

    subgraph WalkForward["3. Walk-Forward Harness"]
        FS --> WFG[Chronological Fold Generator]
        WFG --> TP[Train-Only Scaler / Imputer]
    end

    subgraph Modeling["4. Probabilistic Modeling"]
        TP --> GHMM[Custom Multivariate Gaussian HMM]
        GHMM --> EM[Baum-Welch EM with Multi-Start]
        EM --> MS[Train-Only BIC Selection (K=2..5)]
        MS --> CF[Causal Forward Filter (Online)]
        MS --> SM[Forward-Backward Smoother / Viterbi (Offline)]
    end

    subgraph StrategySim["5. Strategy & Accounting"]
        CF --> SP[State Profiler & Risk Mapper]
        SP --> SIG[Regime Risk Signal (Time t)]
        SIG --> EXEC[Execution Lag & Position Weight (t -> t+1)]
        EXEC --> ACCT[Cost Accounting & Net Returns]
    end

    subgraph Evaluation["6. Diagnostics & Verification"]
        ACCT --> MET[Metrics: CAGR, Sharpe, Drawdown, Turnover]
        ACCT --> BOOT[Block Bootstrap Paired Significance]
        CF --> EVT[Predeclared Event Window Analysis]
        SM --> CMP[Retrospective PELT / MarkovRegression Comparators]
    end

    subgraph Delivery["7. Presentation & Artifacts"]
        MET --> REP[Automated Reports, Model Card, Visualizations]
        REP --> CLI[CLI & Deterministic Demo]
        REP --> API[FastAPI Causal Filtering Endpoint]
    end
```

---

## 2. Subsystem Responsibilities

### 2.1 Data Ingestion & Quality (`regimeflow.data`)
- **Providers:** Implements Yahoo Finance adapter (via `yfinance`) with comprehensive timeout/error handling, alongside a fully deterministic synthetic financial time-series generator.
- **Cache & Manifest:** Local parquet/CSV cache backed by JSON metadata manifests tracking data source, SHA-256 content hashes, retrieval timestamps, symbol, and adjustment settings.
- **Validation:** Enforces positive prices, monotonic timestamp ordering, absence of duplicate timestamps, OHLC consistency ($L_t \le O_t, C_t \le H_t$), and detection of anomalous session gaps.

### 2.2 Feature Engineering (`regimeflow.features`)
- **Backward-Looking Invariant:** All indicators (log returns, realized volatility, volatility ratio, momentum, Garman-Klass range variance, volume z-scores) strictly look backward. No future fills, no centered windows.
- **Warm-Up Management:** Tracks mandatory warm-up periods explicitly and removes warm-up NaNs in a reproducible, audited manner.
- **Train-Only Preprocessing:** Scalers (StandardScaler/RobustScaler) are fit strictly within training window folds and applied frozen to test slices.

### 2.3 Custom Gaussian HMM (`regimeflow.models.gaussian_hmm`)
- **Mathematics From Scratch:** Pure NumPy/SciPy implementation.
  - Log-domain forward recursion with `scipy.special.logsumexp`.
  - Log-domain backward recursion.
  - Causal forward filtering producing $P(z_t \mid x_{1:t})$ and one-step predictive log-likelihood $\log P(x_t \mid x_{1:t-1})$.
  - Retrospective forward-backward smoothing producing $P(z_t \mid x_{1:T})$.
  - Viterbi dynamic programming producing the MAP state path.
  - Log-emission density via Cholesky decomposition ($L L^T = \Sigma_k$) with triangular solves to avoid explicit matrix inversion.
  - Covariance regularization: controlled ridge adjustment ($\Sigma_k + \epsilon I$) and jitter escalation to guarantee symmetric positive-definiteness.
  - Baum-Welch EM: Updates for initial state distribution $\pi$, transition matrix $A$, state means $\mu_k$, and state covariances $\Sigma_k$ (full and diagonal modes).
  - Degeneracy handling: Minimum occupancy thresholds reject collapsing states.

### 2.4 Model Selection & State Alignment (`regimeflow.models.selection`, `regimeflow.states`)
- **BIC Selection:** Calculated strictly on training data using exact parameter counting:
  $$p_{\text{diag}} = K(K-1) + (K-1) + 2Kd$$
  $$p_{\text{full}} = K(K-1) + (K-1) + Kd + \frac{Kd(d+1)}{2}$$
- **State Alignment:** Permutation-aware matching across walk-forward folds via Hungarian algorithm on emission risk profiles (volatility, mean return) to prevent arbitrary integer label permutation.

### 2.5 Walk-Forward Validation Engine (`regimeflow.validation`)
- Chronological, non-overlapping test folds (e.g., 750 sessions train, 126 sessions test).
- Automated causality assertions: Perturbations to test observations after time $t$ strictly cannot alter filtered probabilities or model selection through time $t$.

### 2.6 Strategy Simulation & Accounting (`regimeflow.strategy`)
- **Lag Convention:** Feature $x_t$ observed at close of day $t \implies$ Filtered posterior $P(z_t \mid x_{1:t}) \implies$ Target weight $w_t$ determined $\implies$ Position held for return $r_{t+1}$ from $t$ to $t+1$.
- **Cost Model:** Transaction costs deducted proportionally to turnover:
  $$\text{Cost}_t = c \cdot |w_t - w_{t-1}|$$
- **Baselines:** Passive Buy & Hold, 200-day Simple Moving Average trend filter.

### 2.7 Statistical Significance & Diagnostics (`regimeflow.metrics`)
- Moving-block bootstrap for paired return differences to preserve time-series autocorrelation and heteroskedasticity.
- Predeclared event-window performance analysis (Global Financial Crisis 2007–2009, COVID-19 shock 2020, 2022 Inflation Selloff).

### 2.8 Comparator Layer (`regimeflow.models.reference`, `regimeflow.changepoints`)
- `hmmlearn.GaussianHMM`: Algorithm verification and log-likelihood sanity comparison.
- `ruptures` PELT: Offline changepoint baseline (explicitly labeled non-causal).
- `statsmodels` Markov-switching regression: Univariate regime comparator.
- BOCPD (Bayesian Online Changepoint Detection): Conjugate Normal-Inverse-Gamma online changepoint model.
