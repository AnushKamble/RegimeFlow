# RegimeFlow Model Card

## 1. Model Details
- **Model Name:** RegimeFlow Gaussian Hidden Markov Model
- **Model Type:** Probabilistic Latent State Engine (Discrete-State First-Order Markov Chain with Continuous Multivariate Gaussian Emissions)
- **Version:** 0.1.0
- **Authors:** RegimeFlow Engineering Team
- **License:** MIT
- **Primary Method:** First-principles NumPy/SciPy Baum-Welch Expectation-Maximization with log-domain dynamic programming and Cholesky covariance regularization.

---

## 2. Intended Use
- **Primary Use Case:** Exploratory research and quantitative risk monitoring of statistical regime shifts in financial asset returns and volatility.
- **Intended Users:** Quantitative researchers, machine learning engineers, and time-series developers.
- **Evaluation Mechanism:** Chronological walk-forward validation with strict out-of-sample causal forward filtering.

---

## 3. Prohibited & Out-of-Scope Uses
- **Prohibited:** Live capital deployment without human supervision.
- **Prohibited:** High-frequency intraday trading execution.
- **Prohibited:** Presenting retrospective full-sequence smoothed probabilities or Viterbi paths as real-time predictive signals.
- **Disclaimer:** This software is an academic and engineering research prototype. It does not provide financial or investment advice.

---

## 4. Input Features
The model operates on a compact, backward-looking financial feature set:
1. `log_return`: Daily log return $\log(AdjClose_t / AdjClose_{t-1})$.
2. `realized_vol_21`: 21-session rolling sample standard deviation of log returns, annualized ($\times \sqrt{252}$).
3. `vol_ratio`: Short-term (5-day) to long-term (63-day) realized volatility ratio.
4. `momentum_21` (Extended Group): 21-session trailing return momentum.
5. `garman_klass_5` (Extended Group): 5-day rolling Garman-Klass range-based volatility.
6. `volume_zscore_63` (Extended Group): 63-day trailing log volume z-score.

All features are normalized using fold-local `TrainOnlyStandardScaler` or `TrainOnlyRobustScaler` fit strictly on training windows.

---

## 5. Training & Inference Procedure
- **State Selection:** $K \in \{2, 3, 4, 5\}$ candidate state counts evaluated on training folds using Bayesian Information Criterion (BIC) with exact nominal free parameter counting:
  $$p_{\text{full}} = K(K-1) + (K-1) + Kd + K \frac{d(d+1)}{2}$$
- **Multi-Start EM:** Multiple seeded initializations (quantile-partitioned) to evade sub-optimal local extrema.
- **Covariance Regularization:** Adaptive diagonal ridge ($\Sigma_k + \epsilon I$, $\epsilon \ge 10^{-4}$) with Cholesky factorization.
- **Online Filtering:** State posteriors at time $t$ evaluated strictly using forward recursion:
  $$P(z_t \mid x_{1:t}) \propto p(x_t \mid z_t) \sum_{z_{t-1}} A_{z_{t-1}, z_t} P(z_{t-1} \mid x_{1:t-1})$$

---

## 6. Known Limitations & Failure Modes
1. **Regime Identification Ambiguity:** Hidden states are latent statistical clusters. State IDs do not map inherently to economic concepts like "bull" or "bear" without explicit observable risk mapping.
2. **Transition Lag:** Because the causal forward filter relies on incoming price and range data, rapid exogenous shocks (e.g., flash crashes) require 1–3 sessions of elevated volatility before the posterior shifts decisively.
3. **Turnover & Transaction Costs:** In choppy or mean-reverting regimes, oscillations between calm and high-risk states can generate excessive turnover and erode returns after transaction costs.
4. **Non-Stationarity:** Macroeconomic regimes may shift permanently, altering the true data-generating process beyond historical training windows.
