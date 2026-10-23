"""
Unified experiment runner orchestrating data ingestion, feature engineering,
walk-forward validation, strategy simulation, bootstrap, and artifact exports.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from regimeflow.config import ExperimentConfig
from regimeflow.data.providers import DataPipeline
from regimeflow.features.diagnostics import compute_feature_diagnostics
from regimeflow.features.pipeline import FeaturePipeline
from regimeflow.logging_utils import logger
from regimeflow.metrics.bootstrap import paired_block_bootstrap
from regimeflow.metrics.events import analyze_event_windows
from regimeflow.metrics.financial import compute_financial_metrics
from regimeflow.strategy.backtest import BacktestEngine
from regimeflow.strategy.signals import (
    generate_buy_and_hold_signals,
    generate_moving_average_signals,
    generate_regime_signals,
)
from regimeflow.validation.walk_forward import WalkForwardEvaluator


class ExperimentRunner:
    """
    Executes complete end-to-end research experiment pipeline.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run(
        self,
        data_mode_override: Optional[str] = None,
        profile: str = "full",
    ) -> Path:
        """
        Execute experiment run and persist all machine-readable artifacts.
        """
        # Create unique run directory
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_id = f"{self.config.experiment_name}_{profile}_{timestamp_str}"
        run_dir = Path(self.config.output_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        figures_dir = run_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting experiment run '{run_id}' (profile={profile})...")

        # 1. Config adjustments for profile
        cfg = self.config.model_copy(deep=True)
        if data_mode_override:
            cfg.data.source = data_mode_override

        if profile == "smoke":
            logger.info("Applying smoke test parameters (reduced windows, fast candidates)...")
            cfg.data.synthetic_length = 500
            cfg.walk_forward.train_window = 200
            cfg.walk_forward.test_window = 60
            cfg.walk_forward.step_size = 60
            cfg.model.n_states_candidates = [2, 3]
            cfg.model.n_restarts = 2
            cfg.model.max_iter = 25
            cfg.evaluation.bootstrap_replications = 100

        # Snapshot configuration
        cfg.to_yaml(run_dir / "config_snapshot.yaml")

        # 2. Ingest data
        pipeline = DataPipeline()
        raw_df, manifest, quality_report = pipeline.load_or_fetch(cfg.data)
        with open(run_dir / "data_quality.json", "w", encoding="utf-8") as f:
            json.dump(quality_report.to_dict(), f, indent=2)

        # 3. Construct features
        feat_pipe = FeaturePipeline(cfg.features)
        features, forward_returns = feat_pipe.fit_transform(raw_df)
        feat_diag = compute_feature_diagnostics(features)
        pd.DataFrame(feat_diag["summary_statistics"]).to_csv(run_dir / "feature_summary.csv")

        # 4. Walk-forward chronological evaluation
        evaluator = WalkForwardEvaluator(cfg)
        oos_preds, fold_manifest, fitted_artifacts = evaluator.run(features, forward_returns)

        oos_preds.to_csv(run_dir / "oos_predictions.csv")
        fold_manifest.to_csv(run_dir / "fold_manifest.csv", index=False)

        # Candidate models log
        candidate_rows = []
        for item in fitted_artifacts:
            sum_df = item["selector_summary"]
            sum_df["fold_idx"] = item["fold_idx"]
            candidate_rows.append(sum_df)
        if candidate_rows:
            all_candidates_df = pd.concat(candidate_rows, ignore_index=True)
            all_candidates_df.to_csv(run_dir / "candidate_models.csv", index=False)

        # 5. Generate strategy signals on OOS data
        oos_dates = oos_preds.index
        prices = raw_df.loc[oos_dates, "adj_close"]

        sig_bh = generate_buy_and_hold_signals(oos_dates)
        sig_ma = generate_moving_average_signals(
            raw_df["adj_close"], window=cfg.strategy.moving_average_window
        ).loc[oos_dates]
        sig_regime = generate_regime_signals(
            oos_preds["high_risk_prob"],
            multiplier=cfg.strategy.risk_threshold_multiplier,
            min_weight=cfg.strategy.min_weight,
            max_weight=cfg.strategy.max_weight,
        )

        # 6. Simulate backtests with execution lag and turnover costs
        engine = BacktestEngine(
            transaction_cost_bps=cfg.strategy.transaction_cost_bps,
            cash_return_annual=cfg.strategy.cash_return,
        )
        res_bh = engine.run(sig_bh, oos_preds["forward_asset_return"])
        res_ma = engine.run(sig_ma, oos_preds["forward_asset_return"])
        res_regime = engine.run(sig_regime, oos_preds["forward_asset_return"])

        # Strategy returns table
        strategy_returns_df = pd.DataFrame(
            {
                "return_buy_and_hold": res_bh["net_return"],
                "return_moving_average": res_ma["net_return"],
                "return_regime_aware": res_regime["net_return"],
                "nav_buy_and_hold": res_bh["nav_net"],
                "nav_moving_average": res_ma["nav_net"],
                "nav_regime_aware": res_regime["nav_net"],
                "weight_regime": res_regime["weight"],
                "turnover_regime": res_regime["turnover"],
            },
            index=res_regime.index,
        )
        strategy_returns_df.to_csv(run_dir / "strategy_returns.csv")

        # 7. Financial metrics
        met_bh = compute_financial_metrics(res_bh)
        met_ma = compute_financial_metrics(res_ma)
        met_regime = compute_financial_metrics(res_regime)

        metrics_df = pd.DataFrame(
            [
                {"strategy": "Buy_and_Hold", **met_bh},
                {"strategy": "Moving_Average_200", **met_ma},
                {"strategy": "Regime_Aware_Risk_Control", **met_regime},
            ]
        ).set_index("strategy")
        metrics_df.to_csv(run_dir / "strategy_metrics.csv")

        # 8. Paired circular block bootstrap
        boot_res_vs_bh = paired_block_bootstrap(
            strategy_returns=res_regime["net_return"],
            benchmark_returns=res_bh["net_return"],
            block_length=cfg.evaluation.bootstrap_block_length,
            n_replications=cfg.evaluation.bootstrap_replications,
            seed=cfg.evaluation.bootstrap_seed,
        )
        boot_res_vs_ma = paired_block_bootstrap(
            strategy_returns=res_regime["net_return"],
            benchmark_returns=res_ma["net_return"],
            block_length=cfg.evaluation.bootstrap_block_length,
            n_replications=cfg.evaluation.bootstrap_replications,
            seed=cfg.evaluation.bootstrap_seed,
        )

        boot_records = [
            {
                "comparison": "Regime_vs_BuyAndHold",
                "metric": "Sharpe",
                **boot_res_vs_bh["sharpe_diff"],
            },
            {
                "comparison": "Regime_vs_BuyAndHold",
                "metric": "CAGR",
                **boot_res_vs_bh["cagr_diff"],
            },
            {
                "comparison": "Regime_vs_BuyAndHold",
                "metric": "MaxDrawdown",
                **boot_res_vs_bh["max_dd_diff"],
            },
            {
                "comparison": "Regime_vs_MovingAverage",
                "metric": "Sharpe",
                **boot_res_vs_ma["sharpe_diff"],
            },
            {
                "comparison": "Regime_vs_MovingAverage",
                "metric": "CAGR",
                **boot_res_vs_ma["cagr_diff"],
            },
            {
                "comparison": "Regime_vs_MovingAverage",
                "metric": "MaxDrawdown",
                **boot_res_vs_ma["max_dd_diff"],
            },
        ]
        pd.DataFrame.from_records(boot_records).to_csv(
            run_dir / "bootstrap_intervals.csv", index=False
        )

        # 9. Event window study
        event_df = analyze_event_windows(oos_preds, cfg.evaluation.event_windows)
        event_df.to_csv(run_dir / "event_window_metrics.csv", index=False)

        # 10. Run manifest
        run_manifest = {
            "run_id": run_id,
            "experiment_name": cfg.experiment_name,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "profile": profile,
            "data_source": cfg.data.source,
            "symbol": cfg.data.symbol,
            "data_sha256": manifest.sha256_hash,
            "date_range": [manifest.start_date, manifest.end_date],
            "total_raw_sessions": len(raw_df),
            "total_oos_sessions": len(oos_preds),
            "n_folds": len(fold_manifest),
            "artifacts": {
                "config": str(run_dir / "config_snapshot.yaml"),
                "data_quality": str(run_dir / "data_quality.json"),
                "feature_summary": str(run_dir / "feature_summary.csv"),
                "fold_manifest": str(run_dir / "fold_manifest.csv"),
                "candidate_models": str(run_dir / "candidate_models.csv"),
                "oos_predictions": str(run_dir / "oos_predictions.csv"),
                "strategy_returns": str(run_dir / "strategy_returns.csv"),
                "strategy_metrics": str(run_dir / "strategy_metrics.csv"),
                "bootstrap_intervals": str(run_dir / "bootstrap_intervals.csv"),
                "event_window_metrics": str(run_dir / "event_window_metrics.csv"),
            },
        }
        with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(run_manifest, f, indent=2)

        logger.info(f"Experiment run completed successfully. Outputs saved in: {run_dir}")
        return run_dir
