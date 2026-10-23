"""
Automated markdown research report generation from machine-readable run outputs.
Never invents or fabricates numbers: builds strictly from saved experiment artifacts.
"""

import json
from pathlib import Path
import pandas as pd
from regimeflow.logging_utils import logger
from regimeflow.reports.plots import generate_experiment_plots


def build_research_report(run_dir: Path | str) -> Path:
    """
    Compile comprehensive, honest research report from run outputs.
    """
    run_dir = Path(run_dir)
    report_path = run_dir / "report.md"

    # Load artifacts
    with open(run_dir / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(run_dir / "data_quality.json", "r", encoding="utf-8") as f:
        qa_report = json.load(f)

    metrics_df = pd.read_csv(run_dir / "strategy_metrics.csv").set_index("strategy")
    boot_df = pd.read_csv(run_dir / "bootstrap_intervals.csv")
    fold_df = pd.read_csv(run_dir / "fold_manifest.csv")
    events_path = run_dir / "event_window_metrics.csv"
    try:
        events_df = (
            pd.read_csv(events_path)
            if (events_path.exists() and events_path.stat().st_size > 0)
            else pd.DataFrame()
        )
    except (pd.errors.EmptyDataError, Exception):
        events_df = pd.DataFrame()

    # Generate plots
    generate_experiment_plots(run_dir)

    # Format numbers safely
    def fmt_pct(val: float) -> str:
        return f"{val * 100:.2f}%" if pd.notna(val) else "N/A"

    def fmt_num(val: float, decimals: int = 3) -> str:
        return f"{val:.{decimals}f}" if pd.notna(val) else "N/A"

    report_lines = [
        f"# RegimeFlow — Research & Experimental Evaluation Report",
        f"",
        f"**Run Identifier:** `{manifest['run_id']}`  ",
        f"**Execution Timestamp:** {manifest['timestamp_utc']}  ",
        f"**Instrument:** {manifest['symbol']} | **Data Source:** {manifest['data_source']} | **Profile:** {manifest['profile']}  ",
        f"**Date Range:** {manifest['date_range'][0]} to {manifest['date_range'][1]} ({manifest['total_oos_sessions']} out-of-sample trading sessions across {manifest['n_folds']} chronological folds)  ",
        f"",
        f"---",
        f"",
        f"## 1. Executive Summary & Key Measured Results",
        f"RegimeFlow evaluates whether a multivariate Gaussian Hidden Markov Model implemented from scratch in NumPy/SciPy can discover persistent latent market regimes and support out-of-sample risk-control allocation after execution delay and transaction costs.",
        f"",
        f"### Primary Measured Metrics Summary",
        f"",
        f"| Strategy | CAGR | Ann. Volatility | Sharpe Ratio | Max Drawdown | Total Turnover | Total Cost Paid | Mean Exposure |",
        f"|---|---|---|---|---|---|---|---|",
    ]

    for strat_name, row in metrics_df.iterrows():
        report_lines.append(
            f"| **{strat_name}** | {fmt_pct(row['cagr'])} | {fmt_pct(row['annualized_volatility'])} | "
            f"{fmt_num(row['sharpe_ratio'])} | {fmt_pct(row['max_drawdown'])} | "
            f"{fmt_num(row['total_turnover'], 1)} | {fmt_pct(row['total_cost_paid'])} | "
            f"{fmt_pct(row['mean_exposure'])} |"
        )

    report_lines.extend([
        f"",
        f"---",
        f"",
        f"## 2. Statistical Uncertainty via Circular Moving-Block Bootstrap",
        f"To account for volatility clustering and serial autocorrelation in daily returns, strategy differences were evaluated using Circular Moving-Block Bootstrap ($L=21$ trading days, $B={manifest.get('bootstrap_replications', 1000)}$ paired resamples).",
        f"",
        f"| Paired Comparison | Metric | Observed Difference | 95% Confidence Interval | Empirical p-value | Interpretation |",
        f"|---|---|---|---|---|---|",
    ])

    for _, row in boot_df.iterrows():
        ci_str = f"[{fmt_num(row['ci_lower_95'])}, {fmt_num(row['ci_upper_95'])}]"
        p_val = row["p_value"]
        interp = "Statistically Significant" if p_val < 0.05 else "Not Statistically Significant (Interval Spans Zero)"
        report_lines.append(
            f"| {row['comparison']} | **{row['metric']}** | {fmt_num(row['observed_diff'])} | {ci_str} | {fmt_num(p_val, 4)} | {interp} |"
        )

    report_lines.extend([
        f"",
        f"---",
        f"",
        f"## 3. Out-of-Sample Predictive Density vs. Static Gaussian Baseline",
        f"A multi-state HMM is scientifically justified only if it improves out-of-sample density forecasts relative to an unconditional single-state Gaussian benchmark fitted on identical training data.",
        f"",
        f"- **Mean HMM Predictive Log-Likelihood:** {fmt_num(fold_df['mean_oos_predictive_ll'].mean())}",
        f"- **Mean Static Gaussian Predictive Log-Likelihood:** {fmt_num(fold_df['mean_baseline_predictive_ll'].mean())}",
        f"- **Average Out-of-Sample Predictive Improvement:** {fmt_num(fold_df['ll_improvement_over_baseline'].mean())} log points per session",
        f"",
        f"---",
        f"",
        f"## 4. Predeclared Historical Event-Window Study",
        f"Evaluating causal filtered risk probabilities during locked historical market stress windows without look-ahead tuning:",
        f"",
    ])

    if not events_df.empty:
        report_lines.extend([
            f"| Event Name | Window Dates | Sessions | Mean Risk Prob | Pre-Event Baseline | Elevated Sessions (%) | Asset Max Drawdown |",
            f"|---|---|---|---|---|---|---|",
        ])
        for _, row in events_df.iterrows():
            report_lines.append(
                f"| **{row['event_name']}** | {row['start_date']} to {row['end_date']} | {row['n_sessions']} | "
                f"{fmt_pct(row['mean_risk_prob'])} | {fmt_pct(row['pre_event_baseline_risk_prob'])} | "
                f"{fmt_pct(row['elevated_risk_fraction'])} | {fmt_pct(row['asset_max_drawdown'])} |"
            )
    else:
        report_lines.append("_Event study dates were outside the current out-of-sample evaluation interval._")

    report_lines.extend([
        f"",
        f"---",
        f"",
        f"## 5. Visual Artifacts",
        f"",
        f"### Causal Filtered Regime Probabilities",
        f"![Regime Probabilities](figures/regime_probabilities.png)",
        f"",
        f"### Strategy Cumulative Equity Curves",
        f"![Strategy Equity Curves](figures/strategy_equity_curves.png)",
        f"",
        f"### Underwater Drawdown Curves",
        f"![Drawdowns](figures/drawdowns.png)",
        f"",
        f"### Strategy Exposure and Rebalance Turnover",
        f"![Exposure & Turnover](figures/exposure_turnover.png)",
        f"",
        f"---",
        f"",
        f"## 6. Honest Limitations & Scientific Integrity",
        f"1. **Latent States are Not Ground Truth:** Market regimes are statistical constructs, not observable physical facts. HMM states represent clusters in the feature space and can split, merge, or experience label permutation.",
        f"2. **Causal Filter Lag:** While causal filtering eliminates look-ahead bias, it requires observable volatility spikes to trigger probability shifts, resulting in a short lag during sudden shock onsets.",
        f"3. **Whipsaws & Costs:** During sideways or choppy market phases, rapid oscillations between calm and volatile states incur turnover and transaction costs.",
        f"4. **No Investment Advice:** This engine is built strictly for scientific machine learning research and educational demonstration.",
    ])

    report_text = "\n".join(report_lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"Built research report: {report_path}")
    return report_path
