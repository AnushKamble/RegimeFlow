"""
Command-line interface for RegimeFlow.
Provides complete workflow execution from config to report.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from regimeflow.config import ExperimentConfig
from regimeflow.data.providers import DataPipeline
from regimeflow.data.synthetic import generate_synthetic_hmm_data
from regimeflow.experiment import ExperimentRunner
from regimeflow.logging_utils import logger
from regimeflow.models.gaussian_hmm import GaussianHMM
from regimeflow.reports.build_report import build_research_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="regimeflow",
        description="RegimeFlow: Probabilistic Market Regime Detection & Changepoint Engine",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. validate-synthetic / validate-hmm
    val_parser = subparsers.add_parser(
        "validate-synthetic",
        aliases=["validate-hmm"],
        help="Validate custom HMM against synthetic ground-truth sequences",
    )
    val_parser.add_argument("--config", type=str, default="configs/default.yaml")
    val_parser.add_argument("--synthetic", action="store_true", default=True)

    # 2. download-data
    down_parser = subparsers.add_parser(
        "download-data", help="Download and cache historical market data"
    )
    down_parser.add_argument("--config", type=str, default="configs/default.yaml")
    down_parser.add_argument("--force", action="store_true", help="Force re-download")

    # 3. run-experiment / run
    run_parser = subparsers.add_parser(
        "run-experiment", aliases=["run"], help="Execute walk-forward experiment"
    )
    run_parser.add_argument("--config", type=str, default="configs/default.yaml")
    run_parser.add_argument(
        "--data-mode", choices=["yfinance", "cached", "synthetic"], default=None
    )
    run_parser.add_argument(
        "--profile", choices=["smoke", "full"], default="full"
    )

    # 4. make-report / report
    rep_parser = subparsers.add_parser(
        "make-report", aliases=["report"], help="Generate figures and report from run artifacts"
    )
    rep_parser.add_argument("--run-dir", type=str, required=True, help="Path to experiment run directory")

    # 5. demo
    demo_parser = subparsers.add_parser(
        "demo", help="Run 60-second end-to-end demonstration"
    )
    demo_parser.add_argument("--synthetic", action="store_true", default=True)

    # 6. dashboard / serve
    dash_parser = subparsers.add_parser(
        "dashboard", aliases=["serve"], help="Launch interactive visual web dashboard"
    )
    dash_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    dash_parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    dash_parser.add_argument("--run-dir", type=str, default=None, help="Specific experiment run directory to visualize")

    return parser


def handle_validate_synthetic(args: argparse.Namespace) -> int:
    logger.info("Executing synthetic HMM validation...")
    pi = np.array([0.7, 0.3])
    A = np.array([[0.95, 0.05], [0.10, 0.90]])
    means = np.array([[2.0, 1.0], [-2.0, -1.0]])
    covs = np.array([[[0.2, 0.0], [0.0, 0.2]], [[0.3, 0.0], [0.0, 0.3]]])

    X, true_states = generate_synthetic_hmm_data(600, pi, A, means, covs, seed=42)
    model = GaussianHMM(n_states=2, covariance_type="full", n_restarts=3, max_iter=40, random_state=42)
    model.fit(X)

    path, score = model.decode_viterbi(X)
    acc = max(np.mean(path == true_states), np.mean(path == (1 - true_states)))
    print("=" * 60)
    print("SYNTHETIC HMM VALIDATION RESULTS")
    print(f"Sample length: {len(X)} sessions")
    print(f"Log-Likelihood: {model.final_log_likelihood_:.2f}")
    print(f"State Recovery Accuracy: {acc * 100:.1f}%")
    print("Learned Transition Matrix A:")
    print(np.round(model.A_, 3))
    print("Learned Means:")
    print(np.round(model.means_, 3))
    print("=" * 60)
    return 0


def handle_download_data(args: argparse.Namespace) -> int:
    config = ExperimentConfig.from_yaml(args.config)
    pipeline = DataPipeline()
    df, manifest, report = pipeline.load_or_fetch(config.data, force_refresh=args.force)
    print("=" * 60)
    print("DATA INGESTION & QUALITY REPORT")
    print(f"Symbol: {manifest.symbol} (source: {manifest.source})")
    print(f"Total sessions: {len(df)} ({manifest.start_date} to {manifest.end_date})")
    print(f"SHA-256 Hash: {manifest.sha256_hash}")
    print(f"Quality Gate Status: {'PASSED' if report.is_valid else 'FAILED'}")
    if report.warnings:
        print(f"Warnings: {report.warnings}")
    print("=" * 60)
    return 0 if report.is_valid else 1


def handle_run_experiment(args: argparse.Namespace) -> int:
    config = ExperimentConfig.from_yaml(args.config)
    runner = ExperimentRunner(config)
    run_dir = runner.run(data_mode_override=args.data_mode, profile=args.profile)
    build_research_report(run_dir)
    print("=" * 60)
    print("EXPERIMENT EXECUTION COMPLETED")
    print(f"Output Directory: {run_dir}")
    print(f"Research Report: {run_dir / 'report.md'}")
    print("=" * 60)
    return 0


def handle_make_report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        return 1
    report_path = build_research_report(run_dir)
    print(f"Report generated successfully at: {report_path}")
    return 0


def handle_demo(args: argparse.Namespace) -> int:
    logger.info("Running RegimeFlow 60-second demonstration...")
    config = ExperimentConfig(
        experiment_name="regimeflow_demo",
        data={"source": "synthetic", "synthetic_length": 500, "seed": 42},
        walk_forward={"train_window": 200, "test_window": 60, "step_size": 60},
        model={"n_states_candidates": [2, 3], "n_restarts": 2, "max_iter": 25},
        evaluation={"bootstrap_replications": 100},
    )
    runner = ExperimentRunner(config)
    run_dir = runner.run(profile="smoke")
    build_research_report(run_dir)

    # Print summary table
    metrics_path = run_dir / "strategy_metrics.csv"
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path).set_index("strategy")
        print("\n" + "=" * 70)
        print("DEMO PERFORMANCE SUMMARY (Synthetic Out-of-Sample)")
        print("=" * 70)
        print(metrics_df[["cagr", "annualized_volatility", "sharpe_ratio", "max_drawdown", "total_turnover"]])
        print("=" * 70)
        print(f"Artifacts and Figures saved in: {run_dir}")

    return 0


def handle_dashboard(args: argparse.Namespace) -> int:
    import uvicorn
    from regimeflow.api import create_app

    print("=" * 60)
    print("LAUNCHING REGIMEFLOW INTERACTIVE DASHBOARD")
    print(f"Web Dashboard URL: http://{args.host}:{args.port}/")
    print(f"Swagger API Docs:  http://{args.host}:{args.port}/docs")
    print("Press Ctrl+C to terminate the dashboard server.")
    print("=" * 60)

    app = create_app(run_dir=args.run_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def main(args: list[str] | None = None) -> int:
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    if not parsed_args.command:
        parser.print_help()
        return 0

    if parsed_args.command in ["validate-synthetic", "validate-hmm"]:
        return handle_validate_synthetic(parsed_args)
    elif parsed_args.command == "download-data":
        return handle_download_data(parsed_args)
    elif parsed_args.command in ["run-experiment", "run"]:
        return handle_run_experiment(parsed_args)
    elif parsed_args.command in ["make-report", "report"]:
        return handle_make_report(parsed_args)
    elif parsed_args.command == "demo":
        return handle_demo(parsed_args)
    elif parsed_args.command in ["dashboard", "serve"]:
        return handle_dashboard(parsed_args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
