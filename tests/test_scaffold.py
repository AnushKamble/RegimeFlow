"""
Tests for project scaffolding, configuration loading, and CLI invocation.
"""

import subprocess
import sys
import pytest
from regimeflow import __version__
from regimeflow.config import ExperimentConfig, ModelConfig


def test_package_version():
    assert __version__ == "0.1.0"


def test_default_config_loading():
    config = ExperimentConfig.from_yaml("configs/default.yaml")
    assert config.experiment_name == "regimeflow_primary"
    assert config.data.symbol == "SPY"
    assert config.model.n_states_candidates == [2, 3, 4, 5]
    assert config.walk_forward.train_window == 750
    assert config.walk_forward.test_window == 126
    assert config.strategy.transaction_cost_bps == 10.0


def test_invalid_config_raises_error():
    with pytest.raises(Exception):
        ModelConfig(n_states_candidates=[1, 2])


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "regimeflow.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "RegimeFlow" in result.stdout
    assert "validate-synthetic" in result.stdout
    assert "run-experiment" in result.stdout
