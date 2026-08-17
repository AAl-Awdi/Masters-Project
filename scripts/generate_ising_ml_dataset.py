"""Generate the Day 25 classical Ising configuration dataset from the YAML."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ising2d.ml_dataset import (  # noqa: E402
    build_classical_ml_dataset,
    create_validation_figures,
    load_previous_day_seeds,
    require_output_paths_available,
    save_dataset,
    validate_dataset,
    validate_dataset_specification,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retain raw 2D Ising configurations at fixed production-sweep "
            "intervals and save the lean Day 25 dataset outputs."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the Day 25 YAML configuration.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the design without running chains or writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Deliberately replace all declared outputs.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress one-line progress output for each chain.",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_configuration(path: Path) -> tuple[dict[str, Any], Path]:
    absolute_path = project_path(path).resolve()
    if not absolute_path.exists():
        raise FileNotFoundError(f"Configuration not found: {absolute_path}")
    with absolute_path.open("r", encoding="utf-8") as handle:
        specification = yaml.safe_load(handle)
    if not isinstance(specification, dict):
        raise ValueError("The YAML root must be a mapping.")
    validate_dataset_specification(specification)
    return specification, absolute_path


def print_plan(specification: dict[str, Any]) -> None:
    parameters = validate_dataset_specification(specification)
    n_temperatures = int(parameters["temperatures"].size)
    n_replicates = int(parameters["n_replicates"])
    n_chains = n_temperatures * n_replicates
    n_samples = n_chains * int(parameters["configurations_per_chain"])
    lattice_size = int(parameters["lattice_size"])
    proposals = (
        n_chains
        * (
            int(parameters["burn_in_sweeps"])
            + int(parameters["production_sweeps"])
        )
        * lattice_size**2
    )
    raw_bytes = n_samples * lattice_size**2 * np.dtype(np.int8).itemsize

    print(f"Experiment: {specification['experiment']['name']}")
    print(f"Schema version: {specification['schema_version']}")
    print(f"Lattice: {lattice_size} x {lattice_size}")
    print(f"Temperatures: {n_temperatures}")
    print(f"Chains per temperature: {n_replicates}")
    print(f"Total chains: {n_chains}")
    print(
        "Burn-in / production / retention interval: "
        f"{parameters['burn_in_sweeps']} / "
        f"{parameters['production_sweeps']} / "
        f"{parameters['configuration_interval']} sweeps"
    )
    print(
        "Saved configurations per chain: "
        f"{parameters['configurations_per_chain']}"
    )
    print(f"Total saved configurations: {n_samples:,}")
    print(f"Single-spin proposals: {proposals:,}")
    print(
        "Seed range: "
        f"{int(parameters['seed_matrix'].min())} to "
        f"{int(parameters['seed_matrix'].max())}"
    )
    print(f"Raw int8 configuration-array size: {raw_bytes / 1024**2:.2f} MiB")
    print("Outputs:")
    for name, relative_path in specification["outputs"].items():
        print(f"  {name}: {project_path(Path(str(relative_path)))}")


def main() -> int:
    args = parse_arguments()
    specification, _ = load_configuration(args.config)
    print_plan(specification)

    if args.dry_run:
        print("Dry run complete: no chains were run and no files were written.")
        return 0

    require_output_paths_available(
        specification,
        project_root=PROJECT_ROOT,
        overwrite=args.overwrite,
    )

    previous_seeds = load_previous_day_seeds(PROJECT_ROOT / "results" / "raw")
    dataset = build_classical_ml_dataset(
        specification,
        verbose=not args.quiet,
    )
    validation = validate_dataset(
        dataset,
        specification,
        previous_seeds=previous_seeds,
    )
    saved = save_dataset(
        specification,
        dataset,
        project_root=PROJECT_ROOT,
        overwrite=args.overwrite,
    )

    day23_summary = (
        PROJECT_ROOT
        / "results"
        / "processed"
        / "ising_day23_multiseed_aggregate_summary.csv"
    )
    figures, comparison = create_validation_figures(
        specification,
        dataset,
        saved["diagnostics"],
        project_root=PROJECT_ROOT,
        overwrite=args.overwrite,
        day23_summary_path=day23_summary if day23_summary.exists() else None,
    )

    print("\nDataset generation completed cleanly.")
    print(f"Elapsed simulation time: {dataset['elapsed_seconds']:.3f} s")
    print(f"Saved shape: {np.asarray(dataset['configurations']).shape}")
    print(
        "Compressed NPZ size: "
        f"{saved['paths']['dataset_npz'].stat().st_size / 1024**2:.3f} MiB"
    )
    print(f"Integrity checks passed: {validation['all_integrity_checks_passed']}")
    print(f"Both low-temperature sectors: {validation['both_low_temperature_sectors']}")
    if "day23_energy_per_spin" in comparison:
        max_energy_difference = float(
            comparison["energy_difference_day25_minus_day23"].abs().max()
        )
        max_magnetisation_difference = float(
            comparison[
                "absolute_magnetisation_difference_day25_minus_day23"
            ].abs().max()
        )
        print(
            "Largest absolute Day 25-Day 23 differences: "
            f"energy={max_energy_difference:.6f}, "
            f"|m|={max_magnetisation_difference:.6f}"
        )
    print("Saved files:")
    for name, path in saved["paths"].items():
        if name != "figure_dir":
            print(f"  {name}: {path}")
    for name, path in figures.items():
        print(f"  figure_{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
