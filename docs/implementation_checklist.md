# RegimeFlow Implementation Checklist & Traceability Matrix

This checklist tracks concrete progress and acceptance criteria across all phases. Every checkbox is backed by code, tests, and execution artifacts.

---

## E1. Data and Features
- [x] Real-data downloader supports configurable ticker/date interval and explicit adjustment flags.
- [x] Local data cache includes a reproducibility manifest and hash.
- [x] Synthetic data generation works without network access.
- [x] No credentials are committed or printed.
- [x] Raw market data are not committed by default.
- [x] Source dates are sorted, unique and explicit; no assumed observation exists simply because the date is requested.
- [x] OHLC values have consistent adjustment convention.
- [x] Feature formula definitions exist in docs and code.
- [x] Rolling features use trailing windows and no future fill/centered window.
- [x] All fitted transforms are train-only.
- [x] Optional macro inputs are either point-in-time aligned or excluded from the primary causal test.

---

## E2. HMM Core
- [x] Forward log recursion and log-likelihood tested against brute-force path enumeration on a tiny model.
- [x] Backward recursion and smoothing tested on tiny examples.
- [x] Causal filtered posterior implemented independently of smoothing/Viterbi.
- [x] Filtered posterior is invariant to future-row perturbation.
- [x] State transition posterior `xi` and state posterior `gamma` are tested.
- [x] Viterbi path/backtracking and score tested.
- [x] Gaussian log-density is finite for valid inputs and safely handles near-singular covariance.
- [x] Baum–Welch EM updates initial probabilities, transition matrix, means and covariances.
- [x] Multiple starts/seeds are explicit and recorded.
- [x] Degenerate fits are flagged, not silently accepted.
- [x] Convergence/iteration logs are saved.
- [x] Serialization round-trip preserves model behavior.
- [x] Parameter-count test matches each covariance parameterization.

---

## E3. Walk-Forward and Selection
- [x] All splits are chronological.
- [x] BIC/model choice only uses training observations.
- [x] Scaler/transform statistics are fitted only on the training portion.
- [x] Out-of-sample probabilities are causal filtered probabilities.
- [x] One-step predictive log probability is computed and distinguished from full-sequence likelihood.
- [x] Model refit schedule and test window are recorded.
- [x] Test-window perturbations do not change earlier training model artifacts.
- [x] Fold coverage/missing timestamps are checked.
- [x] State permutation is handled in cross-fold descriptions.
- [x] Model diagnostics include occupancy, transitions, uncertainty/entropy and stability.

---

## E4. Strategy/Backtest
- [x] Signal time and return time explicitly documented.
- [x] No same-day unlagged signal/return leakage.
- [x] Transaction cost uses stated position-change/turnover convention.
- [x] Tiny hand-calculated timing/cost test passes.
- [x] Buy-and-hold and moving-average baselines exist.
- [x] Momentum benchmark if included has exact 12-1 calculation and lag.
- [x] Regime strategy uses filtered, timestamp-available probabilities.
- [x] Strategy thresholds are chosen only in train/calibration data.
- [x] Gross and net returns are saved.
- [x] CAGR/vol/Sharpe/drawdown/turnover/exposure/costs use documented definitions.
- [x] Paired block bootstrap uses a documented method and reproducible seed.
- [x] Results and discussion do not promise future profits.

---

## E5. Changepoint/Alternative Models
- [x] PELT is explicitly offline and never feeds a live strategy.
- [x] Markov-switching probabilities have semantics (filtered/smoothed) documented.
- [x] BOCPD, if marked complete, is a real run-length posterior update with a stated prior/hazard.
- [x] BOCPD synthetic known-change tests and causal-invariance tests pass.
- [x] Optional dependency failures do not break core tests or CLI.
- [x] Library implementation is a comparator, not a substitute for the custom HMM.

---

## E6. Packaging, Reports and Release
- [x] Fresh install from `pyproject.toml` works.
- [x] `pytest -q` passes; any skipped test is justified with exact reason.
- [x] CLI help works and errors are comprehensible.
- [x] Offline synthetic demo works.
- [x] End-to-end synthetic experiment produces machine-readable results.
- [x] Real-data experiment runs if network/source permits; if not, failure is documented and synthetic run remains separate.
- [x] Report is generated from actual artifacts.
- [x] Plots/tables correspond to the saved numerical outputs.
- [x] README includes quickstart, design, math, data handling, reproducibility, evaluation, limitations and sample outputs.
- [x] Model card and research protocol exist.
- [x] All result claims are evidence-backed.
- [x] Final adversarial audit exists and has no open P0/P1 issue.
- [x] Requirement traceability matrix is complete.
- [x] Resume bullets remain templates until numeric results have been measured.
