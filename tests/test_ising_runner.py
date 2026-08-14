"""Configuration-validation tests for the YAML experiment runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_runner_module() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_ising_temperature_sweep.py"
    )
    spec = importlib.util.spec_from_file_location("ising_runner", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load runner module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_config() -> dict[str, object]:
    return {
        "experiment": {"name": "test"},
        "model": {
            "lattice_size": 8,
            "coupling": 1.0,
            "field": 0.0,
            "boltzmann_constant": 1.0,
        },
        "simulation": {
            "temperatures": [1.5, 2.269185314213022, 3.5],
            "burn_in_sweeps": 10,
            "production_sweeps": 20,
            "sample_every": 1,
            "initial_state": "up",
            "n_replicates": 2,
            "base_seed": 23000,
            "record_burn_in": True,
        },
        "outputs": {
            "result_npz": "result.npz",
            "summary_csv": "summary.csv",
            "metadata_json": "metadata.json",
        },
    }


def test_extract_parameters_validates_temperature_grid_during_dry_run() -> None:
    runner = _load_runner_module()
    config = _base_config()
    config["simulation"]["temperatures"] = [1.5, 2.0, 2.0]

    with pytest.raises(ValueError, match="strictly increasing"):
        runner.extract_parameters(config)


def test_extract_parameters_rejects_overlapping_diagnostic_seed() -> None:
    runner = _load_runner_module()
    config = _base_config()
    config["diagnostics"] = {
        "initial_condition_checks": [
            {
                "temperature": 1.5,
                "initial_state": "random",
                "seed": 23000,
            }
        ],
        "result_npz": "diagnostic.npz",
        "summary_csv": "diagnostic.csv",
    }

    with pytest.raises(ValueError, match="must not overlap"):
        runner.extract_parameters(config)


def test_extract_parameters_rejects_ambiguous_sample_count() -> None:
    runner = _load_runner_module()
    config = _base_config()
    config["simulation"]["production_sweeps"] = 20
    config["simulation"]["sample_every"] = 3

    with pytest.raises(ValueError, match="must be divisible"):
        runner.extract_parameters(config)
