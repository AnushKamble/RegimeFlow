# Decision Record 0001: Core Probabilistic Modeling, Causal Inference, and Strategy Timing Conventions

**Date:** 2026-10-23  
**Status:** Accepted  
**Deciders:** Senior ML Engineer, Quantitative Research Engineer  

---

## 1. Context
Financial regime modeling requires rigorous statistical mechanics and temporal discipline. Many commercial or academic demonstrations compromise rigor by:
1. Using third-party wrappers (such as `hmmlearn`) without understanding internal inference or convergence assumptions.
2. Using full-sequence Viterbi paths or smoothed posteriors $P(z_t \mid x_{1:T})$ to backtest historical trading rules, introducing major look-ahead bias.
3. Using inaccurate BIC formulas that omit initial probabilities or miscount covariance parameters.
4. Aligning trading signals with the contemporary return, earning price moves that occurred prior to decision execution.

---

## 2. Decisions

### Decision 1: Custom First-Principles Gaussian HMM Implementation
- We implement the entire Gaussian Hidden Markov Model from scratch in Python/NumPy/SciPy.
- Dynamic programming forward recursion, backward recursion, and Viterbi decoding are performed strictly in the log domain using `scipy.special.logsumexp` to eliminate floating-point underflow.
- Multivariate Gaussian emission log-densities are evaluated using Cholesky factorizations ($L L^T = \Sigma_k$) with triangular solves $\mathcal{O}(d^2)$ rather than matrix inversion $\Sigma_k^{-1}$.
- `hmmlearn` is retained strictly as an independent comparator to verify parameter recovery and likelihood correctness on controlled synthetic benchmarks.

### Decision 2: Strict Causal Online Filtering vs. Retrospective Decoding
- The primary online inference method is the causal forward filter:
  $$P(z_t = k \mid x_{1:t}) = \frac{p(x_t \mid z_t = k) \sum_{j} A_{jk} P(z_{t-1} = j \mid x_{1:t-1})}{\sum_{m} p(x_t \mid z_t = m) \sum_{j} A_{jm} P(z_{t-1} = j \mid x_{1:t-1})}$$
- Retrospective smoothed posteriors $P(z_t \mid x_{1:T})$ and Viterbi state sequences $\arg\max_z P(z_{1:T} \mid x_{1:T})$ are strictly isolated and labeled as offline analytical tools for diagnostic inspection only.

### Decision 3: Accurate BIC Parameter Counting
For a $K$-state, $d$-dimensional Gaussian HMM with free initial distribution $\pi$, row-stochastic transition matrix $A$, state means $\mu_k$, and state covariance matrices $\Sigma_k$:
- Initial state distribution: $K - 1$ free parameters.
- Transition matrix: $K(K - 1)$ free parameters.
- State means: $K \cdot d$ free parameters.
- State covariances:
  - **Full Covariance Mode:** $K \cdot \frac{d(d + 1)}{2}$ parameters.
    $$p_{\text{full}} = K(K - 1) + (K - 1) + Kd + K \frac{d(d + 1)}{2}$$
  - **Diagonal Covariance Mode:** $K \cdot d$ parameters.
    $$p_{\text{diag}} = K(K - 1) + (K - 1) + 2Kd$$
- Model selection chooses $\arg\min_K \text{BIC}$ where:
  $$\text{BIC} = -2 \log \hat{\mathcal{L}}_{\text{train}} + p \cdot \log(N_{\text{train}})$$

### Decision 4: Strategy Execution Timing and Cost Lag
- Features at session $t$ are constructed from OHLCV up to close $t$.
- Signal $w_t \in [0, 1]$ is generated at close $t$.
- Position is held for subsequent session $t+1$, earning return $r_{t+1} = \log(AdjClose_{t+1} / AdjClose_t)$.
- Transaction cost rate $c$ (default $10 \text{ bps} = 0.001$) is deducted based on absolute weight turnover:
  $$\text{Cost}_t = c \cdot |w_t - w_{t-1}|$$

### Decision 5: Non-IID Resampling for Statistical Significance
- Strategy return differences are evaluated using circular moving-block bootstrap (block size $L = 21$ trading days) over $B = 1,000$ paired replications with fixed random seeds.

---

## 3. Consequences
- **Positive:** Mathematically unassailable research design. Zero look-ahead bias in online predictions or backtests. Transparent, verifiable statistical mechanics.
- **Negative:** Increased implementation complexity. Custom numerical algorithms must be extensively tested against edge cases (singularities, near-empty states, overflow).
