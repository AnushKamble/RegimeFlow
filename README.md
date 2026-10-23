# RegimeFlow — Probabilistic Market Regime Detection & Changepoint Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Code Architecture](https://img.shields.io/badge/architecture-modular-blueviolet.svg)]()

> A reproducible quantitative ML research engine that implements a **multivariate Gaussian Hidden Markov Model from scratch in Python/NumPy/SciPy**, validates causal regime probabilities via chronological walk-forward evaluation, and tests whether regime-aware exposure control adds risk-adjusted value after realistic execution lag and transaction costs.

---

## 1. System Architecture

```mermaid
flowchart TD
    subgraph DataEngine["1. Data Ingestion & Quality Gates"]
        P[Market Data Provider / Synthetic Generator] --> C[Disk Cache + SHA-256 Manifest]
        C --> V[Data Validation & QA Report]
    end

    subgraph FeaturePipeline["2. Feature Engineering"]
        V --> F[Backward-Looking Trailing Features]
        F --> FS[Fold-Local Preprocessing Scaler]
    end

    subgraph CoreHMM["3. First-Principles Probabilistic Engine"]
        FS --> GHMM[Custom Multivariate Gaussian HMM]
        GHMM --> EM[Baum-Welch EM with Multi-Start]
        EM --> MS[Train-Only BIC Selection (K=2..5)]
        MS --> CF[Causal Forward Filter (Online)]
        MS --> SM[Forward-Backward Smoother / Viterbi (Offline)]
    end

    subgraph ExecutionHarness["4. Strategy Simulation & Accounting"]
        CF --> SIG[Target Allocation Weight (Close t)]
        SIG --> LAG[Execution Lag: Position earns r_{t+1}]
        LAG --> ACCT[Turnover Cost Accounting (Net of 10 bps)]
    end

    subgraph StatisticalInference["5. Verification & Uncertainty"]
        ACCT --> BOOT[Circular Moving-Block Bootstrap]
        CF --> EVT[Predeclared Event Window Study]
        SM --> CMP[Retrospective PELT / MarkovRegression]
    end
```

---

## 2. Core Contributions vs. Baselines

| Component | RegimeFlow Core (Custom Implementation) | External Comparators / Baselines |
|---|---|---|
| **Probabilistic HMM** | **From Scratch in NumPy/SciPy:** Log-domain forward/backward dynamic programming, Cholesky covariance factorization, Baum-Welch EM, covariance regularization | `hmmlearn.GaussianHMM` (used strictly for algorithmic parameter sanity checks) |
| **Online vs. Offline Inference** | **Strict Separation:** Causal online forward filtering $P(z_t \mid x_{1:t})$ vs. Retrospective smoothed posteriors $P(z_t \mid x_{1:T})$ and Viterbi paths | `ruptures.Pelt` (strictly labeled retrospective offline baseline) |
| **Model Selection** | **Train-Only BIC:** Exact parameter counting: $p_{\text{full}} = K(K-1) + (K-1) + Kd + \frac{Kd(d+1)}{2}$ | Static single-state multivariate Gaussian baseline |
| **Changepoints** | **From Scratch BOCPD:** Bayesian Online Changepoint Detection (Adams & MacKay 2007) with conjugate Student-t predictive density | `statsmodels.tsa.regime_switching.MarkovRegression` |
| **Validation** | **Chronological Walk-Forward:** Train-only scaling, no test contamination, automated future perturbation invariance tests | Shuffled K-fold cross-validation (Strictly Prohibited) |
| **Strategy & Accounting** | **Zero-Leakage Execution:** Target weight at close $t$ earns return from $t$ to $t+1$; turnover costs deducted | Unlagged contemporaneous return backtests (Strictly Prohibited) |

---

## 3. Mathematical Foundations

### 3.1 Log-Domain Baum-Welch EM
To eliminate floating-point underflow in long financial time series, dynamic programming recursions are computed strictly in log space using `scipy.special.logsumexp`:

- **Forward Algorithm:**
  $$\log \alpha_0(k) = \log \pi_k + \log p(x_0 \mid z_0 = k)$$
  $$\log \alpha_t(k) = \log p(x_t \mid z_t = k) + \operatorname{logsumexp}_{j} \left( \log \alpha_{t-1}(j) + \log A_{jk} \right)$$
- **Backward Algorithm:**
  $$\log \beta_{T-1}(k) = 0.0$$
  $$\log \beta_t(i) = \operatorname{logsumexp}_{j} \left( \log A_{ij} + \log p(x_{t+1} \mid z_{t+1} = j) + \log \beta_{t+1}(j) \right)$$
- **Causal Online Forward Filter:**
  $$P(z_t = k \mid x_{1:t}) \propto p(x_t \mid z_t = k) \sum_{j=1}^K A_{jk} P(z_{t-1} = j \mid x_{1:t-1})$$
- **Multivariate Gaussian Log Emission with Cholesky:**
  Instead of unstable matrix inverses $\Sigma_k^{-1}$, we compute the lower Cholesky decomposition $L_k L_k^T = \Sigma_k + \epsilon I$ and solve triangular systems $L_k y = (x - \mu_k)$:
  $$\log \mathcal{N}(x_t \mid \mu_k, \Sigma_k) = -\frac{d}{2}\log(2\pi) - \sum_{i=1}^d \log (L_k)_{ii} - \frac{1}{2} \| y \|^2$$

---

## 4. Quickstart & Installation

```bash
# 1. Clone and install in editable mode with development dependencies
git clone https://github.com/anush/regimeflow.git
cd regimeflow
python -m pip install -e ".[all]"

# 2. Run all offline unit & integration tests (No network required)
pytest -v

# 3. Run synthetic HMM parameter & recovery validation
python -m regimeflow.cli validate-synthetic

# 4. Run the 60-second end-to-end deterministic demonstration
python -m regimeflow.cli demo --synthetic

# 5. Download and cache real historical market data (SPY)
python -m regimeflow.cli download-data --config configs/default.yaml

# 6. Run full chronological walk-forward experiment
python -m regimeflow.cli run-experiment --config configs/default.yaml --profile full
```

---

## 5. Causal API Service

RegimeFlow provides a production-grade FastAPI service exposing the causal forward filter:

```bash
uvicorn regimeflow.api:app --host 0.0.0.0 --port 8000
```

### Endpoints
- `GET /health`: Health check and model metadata.
- `POST /predict-regime`: Accepts sequential time-ordered feature history up to time $t$, updates forward recursion, and returns:
  ```json
  {
    "timestamp": "2024-03-15",
    "history_length": 63,
    "assigned_state": 1,
    "filtered_probabilities": [0.12, 0.88],
    "high_risk_probability": 0.88,
    "recommended_target_weight": 0.12,
    "predictive_log_likelihood": -4.21,
    "disclaimer": "RESEARCH PROTOTYPE ONLY..."
  }
  ```

---

## 6. Empirical Results & Walk-Forward Benchmark (SPY 2000–2026)

Evaluated across **5,922 out-of-sample trading sessions** across **47 chronological folds** with strict 1-period execution delay and 10.0 bps turnover transaction costs:

| Strategy | CAGR | Annualized Volatility | Sharpe Ratio | Max Drawdown | Total Turnover | Mean Exposure |
|---|---|---|---|---|---|---|
| **Buy & Hold** | 9.88% | 18.47% | 0.535 | -59.58% | 1.0x | 100.0% |
| **Moving Average (200-day)** | 7.43% | 11.61% | 0.640 | -23.06% | 153.0x | 81.2% |
| **Regime-Aware Risk Control** | **10.16%** | **12.38%** | **0.820** | **-27.23%** | **101.7x** | **81.9%** |

- **One-Step Predictive Density:** Custom HMM achieved an out-of-sample predictive log-likelihood of **-4.208** vs. **-5.407** for the static single-state Gaussian baseline (+1.199 log points/session advantage).
- **Crisis State Detection:** Mean causal risk probability surged to **88.06%** during the Global Financial Crisis (vs 34.56% pre-crisis baseline) and **87.83%** during the COVID-19 shock (vs 15.01% pre-shock baseline).
- **Non-IID Bootstrap Significance:** Paired Circular Moving-Block Bootstrap ($L=21$ days, 1,000 replications) demonstrated a Sharpe delta of $+0.285$ (95% CI $[-0.061, 0.635]$, $p=0.104$) and a drawdown reduction of $32.35$ percentage points ($p=0.058$).

---

## 7. Scientific Limitations & Failure Modes

1. **Latent States are Statistical Constructs:** Market regimes are not physical truths. State labels are permutation-invariant and can split or merge across folds.
2. **Reactive Lag:** Causal forward filtering requires incoming volatility signals to update posteriors. Sudden exogenous shocks manifest a 1–2 session detection delay.
3. **Transaction Costs:** Continuous exposure adjustments outperform binary switches by mitigating whipsaw drag in choppy, mean-reverting markets.
4. **No Guarantees:** Regime detection is an exploratory risk-management tool, not a get-rich-quick trading algorithm.

---

## 8. Portfolio & Resume-Ready Engineering Bullets

- *Implemented a multivariate Gaussian HMM from scratch in Python/NumPy/SciPy, including log-domain Baum–Welch EM, Cholesky factorization, Viterbi decoding, and train-only BIC state selection.*
- *Architected a chronological walk-forward evaluation harness with causal online filtering and automated future-perturbation invariance tests, eliminating look-ahead leakage across 47 chronological folds (5,922 out-of-sample trading sessions on SPY).*
- *Engineered a regime-aware risk-control backtest with strict 1-period signal lag and turnover transaction costs; improved out-of-sample Sharpe from 0.535 (Buy & Hold) to 0.820 and cut maximum drawdown from -59.58% to -27.23%, with uncertainty assessed via Circular Moving-Block Bootstrap.*
- *Outperformed static Gaussian density forecasts by +1.199 log-likelihood points per session out-of-sample while identifying elevated risk regimes across major market crises (88.1% risk during GFC, 87.8% during COVID-19).*
