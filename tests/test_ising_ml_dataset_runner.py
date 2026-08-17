from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "generate_ising_ml_dataset.py"
MAIN_CONFIG = (
    PROJECT_ROOT / "configs" / "ising_day25_classical_ml_dataset_v1.yaml"
)


def test_main_configuration_dry_run_uses_lean_schema() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(MAIN_CONFIG),
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Schema version: 1.0" in completed.stdout
    assert "Total saved configurations: 5,400" in completed.stdout
    assert "Dry run complete" in completed.stdout


def test_smoke_script_writes_only_declared_outputs(tmp_path: Path) -> None:
    specification = {
        "schema_version": "1.0",
        "experiment": {
            "name": "day25_script_smoke",
            "description": "Temporary script smoke test.",
        },
        "model": {
            "lattice_size": 4,
            "coupling": 1.0,
            "field": 0.0,
            "boltzmann_constant": 1.0,
        },
        "dataset": {
            "temperatures": [1.5, 2.269185314213022, 3.5],
            "burn_in_sweeps": 2,
            "production_sweeps": 4,
            "configuration_interval": 2,
            "configurations_per_chain": 2,
            "n_replicates": 2,
            "base_seed": 25950,
            "initial_state_schedule": ["up", "down"],
            "split_schedule": ["train", "test"],
        },
        "outputs": {
            "dataset_npz": str(tmp_path / "dataset.npz"),
            "manifest_csv": str(tmp_path / "manifest.csv"),
            "summary_csv": str(tmp_path / "summary.csv"),
            "chain_diagnostics_csv": str(tmp_path / "diagnostics.csv"),
            "figure_dir": str(tmp_path / "figures"),
        },
    }
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(
        yaml.safe_dump(specification, sort_keys=False), encoding="utf-8"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config_path),
            "--quiet",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "dataset.npz").exists()
    assert (tmp_path / "manifest.csv").exists()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "diagnostics.csv").exists()
    assert len(list((tmp_path / "figures").glob("*.png"))) == 4
    assert not list(tmp_path.glob("*metadata*.json"))
    assert not list(tmp_path.glob("*CARD.md"))
    assert not list(tmp_path.glob("*split_summary*.csv"))
