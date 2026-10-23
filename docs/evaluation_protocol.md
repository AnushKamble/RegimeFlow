# RegimeFlow Evaluation Protocol & Experimental Safeguards

## 1. Principles of Causal Evaluation
Financial time series cannot be evaluated with standard k-fold random cross-validation because shuffling introduces future look-ahead contamination. RegimeFlow strictly implements **chronological walk-forward validation** with absolute temporal boundaries.

---

## 2. Walk-Forward Split Design

### 2.1 Default Window Configuration
- **Training Window:** 750 trading sessions (~3 calendar years).
- **Test Window:** 126 trading sessions (~6 calendar months).
- **Step / Retraining Frequency:** 126 trading sessions (successive, non-overlapping test folds).
- **Split Mechanics:**
  $$\text{Fold } m: \quad \text{Train } [t_{start, m}, t_{train\_end, m}], \quad \text{Test } [t_{test\_start, m}, t_{test\_end, m}]$$
  where $t_{test\_start, m} = t_{train\_end, m} + 1 \text{ session}$.

### 2.2 Leakage Prevention Rules
1. **Preprocessing:** Scaler parameters (mean $\mu_{train}$, standard deviation $\sigma_{train}$, quantile thresholds) are fit exclusively on the training window $[t_{start, m}, t_{train\_end, m}]$ and applied frozen to the test window.
2. **Model Selection:** BIC scores are calculated strictly on training sequence log-likelihood. No test observation is ever evaluated during model selection.
3. **Threshold Calibration:** Risk-state classifications and exposure thresholds ($\lambda, w_{min}, w_{max}$) are derived strictly from training emission distributions.
4. **Causal Forward Filter:** Within the test window, state probabilities are estimated strictly using the forward recursion:
   $$P(z_t \mid x_{1:t}) \propto p(x_t \mid z_t) \sum_{z_{t-1}} A_{z_{t-1}, z_t} P(z_{t-1} \mid x_{1:t-1})$$
   Backward smoothing passes $P(z_t \mid x_{1:T})$ and full-sample Viterbi paths are strictly prohibited from generating trading signals or online evaluation metrics.

---

## 3. One-Step-Ahead Predictive Log-Likelihood

For each test observation $x_t$ ($t \in \text{Test}$), the one-step-ahead predictive likelihood is:
$$p(x_t \mid x_{1:t-1}) = \sum_{k=1}^K p(x_t \mid z_t = k) \, P(z_t = k \mid x_{1:t-1})$$
where the prior predictive distribution for state $z_t$ is:
$$P(z_t = k \mid x_{1:t-1}) = \sum_{j=1}^K A_{jk} \, P(z_{t-1} = j \mid x_{1:t-1})$$
The sequence average predictive log-likelihood:
$$\overline{\mathcal{LL}}_{oos} = \frac{1}{|\text{Test}|} \sum_{t \in \text{Test}} \log p(x_t \mid x_{1:t-1})$$
This is compared directly against the static Gaussian baseline fitted on the same training fold:
$$\log p_{\text{static}}(x_t) = \log \mathcal{N}(x_t \mid \hat{\mu}_{train}, \hat{\Sigma}_{train})$$

---

## 4. Strategy Timing & Transaction Cost Accounting

### 4.1 Timing Convention
To ensure zero look-ahead bias:
1. At the close of trading session $t$, daily OHLCV is observed.
2. Feature vector $x_t$ is computed using data available at or before close $t$.
3. Causal filter outputs posterior $P(z_t \mid x_{1:t})$.
4. Target portfolio weight $w_t \in [0, 1]$ is established.
5. The trade is executed at session $t+1$ (approximated conservatively by earning return $r_{t+1} = \log(AdjClose_{t+1} / AdjClose_t)$ or next open-to-open).
6. **Key Invariant:** Signal $w_t$ determined at $t$ earns return from $t$ to $t+1$. Under no circumstances does $w_t$ earn return $r_t$.

### 4.2 Transaction Cost Model
Portfolio turnover at session $t$ is:
$$\Delta w_t = |w_t - w_{t-1}|$$
With configured cost rate $c$ (default $10 \text{ bps} = 0.0010$):
$$\text{Cost}_t = c \cdot \Delta w_t$$
Net return:
$$R_{net, t+1} = w_t \cdot R_{asset, t+1} - \text{Cost}_t$$

### 4.3 Benchmark Strategies
1. **Buy & Hold (B&H):** $w_t = 1.0$ for all $t$.
2. **200-day Moving Average (200-MA):**
   $$w_t = \begin{cases} 1.0 & \text{if } AdjClose_t \ge \text{SMA}_{200}(AdjClose_t) \\ 0.0 & \text{otherwise} \end{cases}$$
3. **Regime-Aware Risk Control:**
   $$q_t = \sum_{k \in \text{HighRisk}} P(z_t = k \mid x_{1:t}), \quad w_t = \operatorname{clip}(1 - \lambda q_t, 0.0, 1.0)$$

---

## 5. Statistical Significance via Moving-Block Bootstrap

Because daily financial returns exhibit serial dependence, volatility clustering, and fat tails, standard independent identically distributed (IID) bootstrap tests underestimate variance.
- **Method:** Circular Moving-Block Bootstrap (Kunsch 1989; Politis & Romano 1994).
- **Block Length:** Configurable (default $L = 21$ trading days $\approx 1$ calendar month).
- **Replications:** $B = 1,000$ paired resamples.
- **Paired Difference:**
  $$\Delta \text{Metric}^{(b)} = \text{Metric}(\text{RegimeStrategy}^{(b)}) - \text{Metric}(\text{Benchmark}^{(b)})$$
- **Outputs:** Empirical 95% confidence intervals $[q_{0.025}, q_{0.975}]$ and two-sided bootstrap p-values.

---

## 6. Predeclared Historical Stress Windows

The following historical event windows are locked prior to running experiments:
1. **Global Financial Crisis (GFC):** `2007-08-01` to `2009-06-30`
2. **COVID-19 Market Crash:** `2020-02-19` to `2020-04-30`
3. **2022 Inflation & Rate-Hike Drawdown:** `2022-01-03` to `2022-12-30`

Evaluation for each window inspects:
- Speed of transition into elevated-risk state upon volatility spike.
- Dwell time in high-risk state after market troughs.
- Total drawdown reduction vs missed rebound upside.
