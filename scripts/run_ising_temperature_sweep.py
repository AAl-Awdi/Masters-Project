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
    validate_temperature_grid,
)
from ising2d.observables import (  # noqa: E402
    exact_square_lattice_critical_temperature,
)
from ising2d.simulation import run_chain  # noqa: E402


VALID_INITIAL_STATES = {"up", "down", "random"}


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


def _require_integer(
    value: Any,
    *,
    name: str,
    minimum: int,
) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not a Boolean value.")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer.") from error
    if not np.isfinite(numeric_value) or not numeric_value.is_integer():
        raise ValueError(f"{name} must be an integer.")

    integer = int(numeric_value)
    if integer < minimum:
        comparator = "non-negative" if minimum == 0 else f"at least {minimum}"
        raise ValueError(f"{name} must be {comparator}.")
    return integer


def _validate_diagnostic_seeds(
    config: dict[str, Any],
    main_seeds: np.ndarray,
) -> None:
    diagnostics = config.get("diagnostics")
    if not diagnostics:
        return

    checks = diagnostics.get("initial_condition_checks", [])
    diagnostic_seeds: list[int] = []
    for check_index, check in enumerate(checks):
        seed = _require_integer(
            check["seed"],
            name=f"diagnostics.initial_condition_checks[{check_index}].seed",
            minimum=0,
        )
        temperature = float(check["temperature"])
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("diagnostic temperatures must be finite and positive.")

        initial_state = str(check.get("initial_state", "random"))
        if initial_state not in VALID_INITIAL_STATES:
            raise ValueError(
                "diagnostic initial_state must be 'up', 'down' or 'random'."
            )
        diagnostic_seeds.append(seed)

    if len(set(diagnostic_seeds)) != len(diagnostic_seeds):
        raise ValueError("diagnostic chains must use distinct seeds.")
    if np.intersect1d(main_seeds, diagnostic_seeds).size:
        raise ValueError("diagnostic seeds must not overlap the main seed matrix.")


def extract_parameters(config: dict[str, Any]) -> dict[str, Any]:
    model = config["model"]
    simulation = config["simulation"]

    temperatures = validate_temperature_grid(simulation["temperatures"])
    n_replicates = _require_integer(
        simulation["n_replicates"],
        name="simulation.n_replicates",
        minimum=1,
    )
    seeds = make_seed_matrix(
        base_seed=_require_integer(
            simulation["base_seed"],
            name="simulation.base_seed",
            minimum=0,
        ),
        n_replicates=n_replicates,
        n_temperatures=temperatures.size,
    )

    lattice_size = _require_integer(
        model["lattice_size"],
        name="model.lattice_size",
        minimum=2,
    )
    coupling = float(model.get("coupling", 1.0))
    field = float(model.get("field", 0.0))
    boltzmann_constant = float(model.get("boltzmann_constant", 1.0))
    if not np.isfinite(coupling) or coupling <= 0.0:
        raise ValueError("model.coupling must be finite and positive.")
    if not np.isfinite(field):
        raise ValueError("model.field must be finite.")
    if not np.isfinite(boltzmann_constant) or boltzmann_constant <= 0.0:
        raise ValueError(
            "model.boltzmann_constant must be finite and positive."
        )

    burn_in_sweeps = _require_integer(
        simulation["burn_in_sweeps"],
        name="simulation.burn_in_sweeps",
        minimum=0,
    )
    production_sweeps = _require_integer(
        simulation["production_sweeps"],
        name="simulation.production_sweeps",
        minimum=1,
    )
    sample_every = _require_integer(
        simulation.get("sample_every", 1),
        name="simulation.sample_every",
        minimum=1,
    )
    if production_sweeps % sample_every != 0:
        raise ValueError(
            "simulation.production_sweeps must be divisible by "
            "simulation.sample_every."
        )

    initial_state = str(simulation.get("initial_state", "up"))
    if initial_state not in VALID_INITIAL_STATES:
        raise ValueError(
            "simulation.initial_state must be 'up', 'down' or 'random'."
        )

    _validate_diagnostic_seeds(config, seeds)

    return {
        "temperatures": temperatures,
        "seeds": seeds,
        "lattice_size": lattice_size,
        "coupling": coupling,
        "field": field,
        "boltzmann_constant": boltzmann_constant,
        "burn_in_sweeps": burn_in_sweeps,
        "production_sweeps": production_sweeps,
        "sample_every": sample_every,
        "initial_state": initial_state,
        "record_burn_in": bool(simulation.get("record_burn_in", True)),
    }


def _declared_output_paths(config: dict[str, Any]) -> list[Path]:
    outputs = config["outputs"]
    required_output_keys = {"result_npz", "summary_csv", "metadata_json"}
    missing = required_output_keys.difference(outputs)
    if missing:
        raise ValueError(
            "outputs is missing keys: " + ", ".join(sorted(missing))
        )

    paths = [project_path(outputs[key]) for key in sorted(required_output_keys)]
    diagnostics = config.get("diagnostics")
    if diagnostics and diagnostics.get("initial_condition_checks"):
        for key in ("result_npz", "summary_csv"):
            if key not in diagnostics:
                raise ValueError(f"diagnostics is missing key: {key}")
            paths.append(project_path(diagnostics[key]))

    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("all declared result paths must be distinct.")
    return paths


def _refuse_existing_outputs(
    config: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    if overwrite:
        return
    existing = [path for path in _declared_output_paths(config) if path.exists()]
    if existing:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Refusing to start because these outputs already exist:\n"
            f"{formatted}\nPass --overwrite only when replacement is intentional."
        )


def print_plan(config: dict[str, Any], parameters: dict[str, Any]) -> None:
    temperatures = parameters["temperatures"]
    n_replicates = parameters["seeds"].shape[0]
    lattice_size = parameters["lattice_size"]
    total_sweeps = (
        parameters["burn_in_sweeps"] + parameters["production_sweeps"]
    )
    proposal_count = (
        n_replicates * temperatures.size * total_sweeps * lattice_size**2
    )
    exact_tc = exact_square_lattice_critical_temperature(
        coupling=parameters["coupling"],
        boltzmann_constant=parameters["boltzmann_constant"],
    )

    print(f"Experiment: {config['experiment']['name']}")
    print(f"Temperatures: {temperatures.size}")
    print(f"Replicates: {n_replicates}")
    print(f"Lattice: {lattice_size} x {lattice_size}")
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
    print("Outputs:")
    for path in _declared_output_paths(config):
        print(f"  - {path}")


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

    # Fail before a long simulation if any declared destination is already used.
    _refuse_existing_outputs(config, overwrite=args.overwrite)

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
    additional_metadata: dict[str, Any] = {
        "diagnostic_files": diagnostic_files,
    }
    analysis = config.get("analysis", {})
    if analysis.get("notebook"):
        additional_metadata["analysis_notebook"] = str(analysis["notebook"])

    saved = save_temperature_sweep(
        sweep,
        result_path=project_path(outputs["result_npz"]),
        summary_path=project_path(outputs["summary_csv"]),
        metadata_path=project_path(outputs["metadata_json"]),
        config=config,
        config_path=config_path,
        overwrite=args.overwrite,
        additional_metadata=additional_metadata,
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
