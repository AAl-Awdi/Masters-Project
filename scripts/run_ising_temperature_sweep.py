#!/usr/bin/env python3
"""Run a reproducible 2D Ising temperature sweep from a YAML configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Permit direct execution from a fresh checkout before an editable installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ising2d.dataset import (  # noqa: E402
    make_seed_matrix,
    run_temperature_sweep,
    save_chain_collection,
    save_temperature_sweep,
)
from ising2d.observables import (  # noqa: E402
    exact_square_lattice_critical_temperature,
)
from ising2d.simulation import run_chain  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run independent fixed-temperature Metropolis chains and save "
            "raw arrays, a summary CSV and metadata."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML experiment configuration.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs declared by the configuration.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the experiment without running it.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-temperature progress messages.",
    )
    return parser.parse_args()


def load_configuration(path: Path) -> dict[str, Any]:
    absolute_path = path if path.is_absolute() else PROJECT_ROOT / path
    if not absolute_path.exists():
        raise FileNotFoundError(f"Configuration not found: {absolute_path}")

    with absolute_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("The YAML root must be a mapping.")

    required_sections = {"experiment", "model", "simulation", "outputs"}
    missing = required_sections.difference(config)
    if missing:
        raise ValueError(
            "Configuration is missing sections: " + ", ".join(sorted(missing))
        )

    config["_absolute_config_path"] = str(absolute_path.resolve())
    return config


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def extract_parameters(config: dict[str, Any]) -> dict[str, Any]:
    model = config["model"]
    simulation = config["simulation"]

    temperatures = np.asarray(simulation["temperatures"], dtype=float)
    n_replicates = int(simulation["n_replicates"])
    seeds = make_seed_matrix(
        base_seed=int(simulation["base_seed"]),
        n_replicates=n_replicates,
        n_temperatures=temperatures.size,
    )

    parameters = {
        "temperatures": temperatures,
        "seeds": seeds,
        "lattice_size": int(model["lattice_size"]),
        "coupling": float(model.get("coupling", 1.0)),
        "field": float(model.get("field", 0.0)),
        "boltzmann_constant": float(
            model.get("boltzmann_constant", 1.0)
        ),
        "burn_in_sweeps": int(simulation["burn_in_sweeps"]),
        "production_sweeps": int(simulation["production_sweeps"]),
        "sample_every": int(simulation.get("sample_every", 1)),
        "initial_state": str(simulation.get("initial_state", "up")),
        "record_burn_in": bool(simulation.get("record_burn_in", True)),
    }
    return parameters


def print_plan(config: dict[str, Any], parameters: dict[str, Any]) -> None:
    temperatures = parameters["temperatures"]
    n_replicates = parameters["seeds"].shape[0]
    L = parameters["lattice_size"]
    total_sweeps = (
        parameters["burn_in_sweeps"]
        + parameters["production_sweeps"]
    )
    proposal_count = (
        n_replicates * temperatures.size * total_sweeps * L**2
    )
    exact_tc = exact_square_lattice_critical_temperature(
        coupling=parameters["coupling"],
        boltzmann_constant=parameters["boltzmann_constant"],
    )

    print(f"Experiment: {config['experiment']['name']}")
    print(f"Temperatures: {temperatures.size}")
    print(f"Replicates: {n_replicates}")
    print(f"Lattice: {L} x {L}")
    print(
        f"Burn-in / production: {parameters['burn_in_sweeps']} / "
        f"{parameters['production_sweeps']} sweeps"
    )
    print(f"Sample every: {parameters['sample_every']} sweep(s)")
    print(f"Single-spin proposals: {proposal_count:,}")
    print(f"Exact infinite-lattice benchmark: {exact_tc:.9f}")
    print(
        "Seed range: "
        f"{int(parameters['seeds'].min())} to "
        f"{int(parameters['seeds'].max())}"
    )


def run_diagnostics(
    config: dict[str, Any],
    parameters: dict[str, Any],
    *,
    overwrite: bool,
) -> dict[str, str] | None:
    diagnostics = config.get("diagnostics")
    if not diagnostics:
        return None

    checks = diagnostics.get("initial_condition_checks", [])
    if not checks:
        return None

    chains = []
    for check in checks:
        temperature = float(check["temperature"])
        seed = int(check["seed"])
        initial_state = str(check.get("initial_state", "random"))
        print(
            "diagnostic chain: "
            f"T={temperature:.6f}, start={initial_state}, seed={seed}",
            flush=True,
        )
        chains.append(
            run_chain(
                lattice_size=parameters["lattice_size"],
                temperature=temperature,
                burn_in_sweeps=parameters["burn_in_sweeps"],
                production_sweeps=parameters["production_sweeps"],
                sample_every=parameters["sample_every"],
                seed=seed,
                initial_state=initial_state,
                coupling=parameters["coupling"],
                field=parameters["field"],
                boltzmann_constant=parameters["boltzmann_constant"],
                record_burn_in=parameters["record_burn_in"],
            )
        )

    diagnostic_paths = save_chain_collection(
        chains,
        result_path=project_path(diagnostics["result_npz"]),
        summary_path=project_path(diagnostics["summary_csv"]),
        overwrite=overwrite,
    )
    return {name: str(path) for name, path in diagnostic_paths.items()}


def main() -> int:
    args = parse_arguments()
    config = load_configuration(args.config)
    config_path = Path(config.pop("_absolute_config_path"))
    parameters = extract_parameters(config)

    print_plan(config, parameters)
    if args.dry_run:
        print("Dry run complete: no simulations or files were produced.")
        return 0

    sweep = run_temperature_sweep(
        **parameters,
        verbose=not args.quiet,
    )

    diagnostic_files = run_diagnostics(
        config,
        parameters,
        overwrite=args.overwrite,
    )

    outputs = config["outputs"]
    saved = save_temperature_sweep(
        sweep,
        result_path=project_path(outputs["result_npz"]),
        summary_path=project_path(outputs["summary_csv"]),
        metadata_path=project_path(outputs["metadata_json"]),
        config=config,
        config_path=config_path,
        overwrite=args.overwrite,
        additional_metadata={
            "diagnostic_files": diagnostic_files,
            "analysis_notebook": (
                "notebooks/classical/"
                "03_ising_temperature_sweep_thermodynamics.ipynb"
            ),
        },
    )

    print("\nExperiment completed cleanly.")
    print(f"Elapsed time: {float(sweep['elapsed_seconds']):.2f} s")
    for name, path in saved.items():
        print(f"{name}: {path}")
    if diagnostic_files:
        print("diagnostics: " + json.dumps(diagnostic_files, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
