# RegimeFlow Scientific Limitations & Failure Mode Analysis

This document provides a transparent, skeptical audit of the fundamental statistical and practical limitations of the RegimeFlow engine.

---

## 1. Latent States vs. Ground Truth
A Hidden Markov Model (HMM) partitions the continuous observation space into $K$ discrete probability density functions. It does **not** discover an objectively real "market regime":
- **Label Permutation:** State indices ($0, 1, \dots, K-1$) are arbitrary. In successive training folds, the model may swap labels. Canonical alignment (e.g., sorting by volatility) is a necessary post-processing convention, but cannot guarantee identical state semantics across non-stationary periods.
- **State Merging and Splitting:** An increase in candidate state count $K$ from 2 to 3 may split a calm regime into two subtly different low-volatility states, rather than cleanly isolating rare crisis events.

---

## 2. Detection Lag & Reactive Nature
While causal forward filtering strictly prevents look-ahead bias, it has an inherent reactive latency:
- On day 1 of a sudden crisis (e.g., February 24, 2020), the prior predictive probability $P(z_t \mid x_{1:t-1})$ still expects a calm regime due to transition persistence ($A_{00} \approx 0.95$).
- Only after extreme realization $x_t$ occurs does the likelihood $p(x_t \mid z_t = \text{Crisis})$ overwhelm the prior.
- Consequently, the model cannot predict exogenous shocks in advance; it acts as a **reactive risk dampener**, reducing exposure after volatility has already begun to manifest.

---

## 3. Transaction Costs and Whipsaws
- A naive threshold policy (e.g., exit equity if $P(\text{Crisis}) > 0.5$) can suffer severe whipsaws during borderline volatile periods.
- Rebalancing exposure back and forth between 0% and 100% incurs transaction costs (10 bps per unit turnover), which can compound into significant drag.
- In our experiments, continuous clipped allocation ($w_t = \operatorname{clip}(1 - \lambda q_t, 0, 1)$) significantly reduces turnover volatility compared to binary switching.

---

## 4. Model Selection Uncertainty
- BIC penalizes parameter count to avoid overfitting:
  $$\text{BIC} = -2 \log \hat{\mathcal{L}} + p \log N$$
- However, BIC is an asymptotic heuristic that can favor smaller state counts ($K=2$ or $K=3$) in finite samples, even if a 4-state model better captures rare fat-tail events.
- Out-of-sample predictive log-likelihood must be inspected alongside BIC to evaluate generalization performance.

---

## 5. Non-Stationarity and Structural Breaks
- Over decades, market microstructure evolves: trading volume expands, algorithmic market-making increases, and macro policy regimes shift.
- A rolling window of 750 trading sessions (~3 years) adapts to recent distribution shifts but discards deep historical memory from previous business cycles.
