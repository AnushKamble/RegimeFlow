"""
Publication-grade visualizations generated strictly from machine-readable run outputs.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from regimeflow.logging_utils import logger

# Use clean headless backend
plt.switch_backend("Agg")


def generate_experiment_plots(run_dir: Path) -> None:
    """
    Generate all required figures from saved CSV artifacts in run_dir.
    """
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    oos_preds_path = run_dir / "oos_predictions.csv"
    strat_returns_path = run_dir / "strategy_returns.csv"
    fold_manifest_path = run_dir / "fold_manifest.csv"

    if not oos_preds_path.exists() or not strat_returns_path.exists():
        logger.warning("Required CSV outputs missing; skipping plot generation.")
        return

    oos_preds = pd.read_csv(oos_preds_path, parse_dates=["date"]).set_index("date")
    strat_returns = pd.read_csv(strat_returns_path, parse_dates=["date"]).set_index("date")

    # 1. Figure: Filtered Regime Probabilities Timeline
    plt.figure(figsize=(12, 5), dpi=150)
    plt.plot(
        oos_preds.index,
        oos_preds["high_risk_prob"],
        color="#d9534f",
        label="Causal Filtered High-Risk Probability P(z_t = Crisis | x_{1:t})",
        lw=1.2,
    )
    plt.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Risk Threshold (0.5)")
    plt.title("Causal Filtered Regime Probabilities (Out-of-Sample)", fontsize=13, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Probability", fontsize=11)
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(figures_dir / "regime_probabilities.png")
    plt.close()

    # 2. Figure: Strategy Cumulative Equity Curves
    plt.figure(figsize=(12, 6), dpi=150)
    plt.plot(
        strat_returns.index,
        strat_returns["nav_buy_and_hold"],
        label="Buy & Hold (SPY)",
        color="#2c3e50",
        lw=1.5,
    )
    plt.plot(
        strat_returns.index,
        strat_returns["nav_moving_average"],
        label="200-day Moving Average Filter",
        color="#f39c12",
        lw=1.5,
    )
    plt.plot(
        strat_returns.index,
        strat_returns["nav_regime_aware"],
        label="Regime-Aware Risk Control (Net of 10 bps)",
        color="#27ae60",
        lw=1.8,
    )
    plt.title("Out-of-Sample Strategy Cumulative Wealth (Net of Costs)", fontsize=13, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Portfolio NAV (Normalized)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(figures_dir / "strategy_equity_curves.png")
    plt.close()

    # 3. Figure: Underwater Drawdown Curves
    plt.figure(figsize=(12, 5), dpi=150)
    for col, name, color in [
        ("nav_buy_and_hold", "Buy & Hold", "#2c3e50"),
        ("nav_moving_average", "200-MA", "#f39c12"),
        ("nav_regime_aware", "Regime Risk Control", "#27ae60"),
    ]:
        nav = strat_returns[col].to_numpy()
        running_max = np.maximum.accumulate(nav)
        dd = (nav - running_max) / running_max
        plt.plot(strat_returns.index, dd * 100, label=name, color=color, lw=1.2)

    plt.title("Underwater Portfolio Drawdown Comparison", fontsize=13, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Drawdown (%)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(figures_dir / "drawdowns.png")
    plt.close()

    # 4. Figure: Exposure and Turnover
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, dpi=150)
    ax1.plot(
        strat_returns.index,
        strat_returns["weight_regime"],
        color="#2980b9",
        lw=1.2,
        label="Target Exposure w_t",
    )
    ax1.set_ylabel("Portfolio Weight", fontsize=10)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")
    ax1.set_title("Regime-Aware Strategy Exposure & Rebalance Turnover", fontsize=12, fontweight="bold")

    ax2.bar(
        strat_returns.index,
        strat_returns["turnover_regime"],
        color="#e74c3c",
        width=2.0,
        alpha=0.7,
        label="Daily Turnover |w_t - w_{t-1}|",
    )
    ax2.set_ylabel("Turnover", fontsize=10)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(figures_dir / "exposure_turnover.png")
    plt.close()

    # 5. Figure: BIC Candidate Selection across Folds
    if fold_manifest_path.exists():
        folds_df = pd.read_csv(fold_manifest_path)
        plt.figure(figsize=(9, 4), dpi=150)
        plt.bar(
            folds_df["fold_idx"] + 1,
            folds_df["selected_k"],
            color="#8e44ad",
            alpha=0.8,
            width=0.5,
        )
        plt.title("Selected Regime State Count K across Walk-Forward Folds", fontsize=12, fontweight="bold")
        plt.xlabel("Fold Number", fontsize=10)
        plt.ylabel("Selected K (via BIC)", fontsize=10)
        plt.yticks([2, 3, 4, 5])
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(figures_dir / "bic_model_selection.png")
        plt.close()

    logger.info(f"Generated 5 publication figures in {figures_dir}")
