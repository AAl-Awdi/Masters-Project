"""Temperature-sweep orchestration and portable result serialisation."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from .observables import exact_square_lattice_critical_temperature
from .simulation import run_chain


Array = np.ndarray


def validate_temperature_grid(temperatures: Iterable[float]) -> Array:
    """Return a strictly increasing, finite, positive temperature array."""
    values = np.asarray(list(temperatures), dtype=float)

    if values.ndim != 1 or values.size == 0:
        raise ValueError("temperatures must be a non-empty one-dimensional grid.")
    if not np.all(np.isfinite(values)):
        raise ValueError("temperatures must contain only finite values.")
    if np.any(values <= 0.0):
        raise ValueError("all temperatures must be positive.")
    if not np.all(np.diff(values) > 0.0):
        raise ValueError("temperatures must be strictly increasing and unique.")

    return values


def make_seed_matrix(
    base_seed: int,
    n_replicates: int,
    n_temperatures: int,
) -> Array:
    """Assign one distinct integer seed to every replicate-temperature run."""
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative.")
    if n_replicates < 1:
        raise ValueError("n_replicates must be at least one.")
    if n_temperatures < 1:
        raise ValueError("n_temperatures must be at least one.")

    seeds = base_seed + np.arange(
        n_replicates * n_temperatures,
        dtype=np.int64,
    )
    return seeds.reshape(n_replicates, n_temperatures)


def _stack_field(runs: list[list[dict[str, object]]], field: str) -> Array:
    """Stack one chain field into replicate-by-temperature array form."""
    return np.stack(
        [
            np.stack(
                [np.asarray(run[field]) for run in replicate_runs],
                axis=0,
            )
            for replicate_runs in runs
        ],
        axis=0,
    )


def run_temperature_sweep(
    *,
    temperatures: Iterable[float],
    seeds: Array,
    lattice_size: int,
    coupling: float,
    field: float,
    boltzmann_constant: float,
    burn_in_sweeps: int,
    production_sweeps: int,
    sample_every: int,
    initial_state: str,
    record_burn_in: bool = True,
    verbose: bool = True,
) -> dict[str, object]:
    """Run independent fixed-temperature chains over one temperature grid."""
    temperature_grid = validate_temperature_grid(temperatures)
    seed_matrix = np.asarray(seeds, dtype=np.int64)

    if seed_matrix.ndim == 1:
        seed_matrix = seed_matrix[np.newaxis, :]
    if seed_matrix.ndim != 2:
        raise ValueError(
            "seeds must have shape (n_replicates, n_temperatures)."
        )
    if seed_matrix.shape[1] != temperature_grid.size:
        raise ValueError("the seed matrix needs one column per temperature.")
    if np.unique(seed_matrix).size != seed_matrix.size:
        raise ValueError("every chain must have a distinct seed.")

    n_replicates, n_temperatures = seed_matrix.shape
    runs: list[list[dict[str, object]]] = []
    start = perf_counter()

    for replicate_index in range(n_replicates):
        replicate_runs: list[dict[str, object]] = []

        for temperature_index, temperature in enumerate(temperature_grid):
            seed = int(seed_matrix[replicate_index, temperature_index])
            if verbose:
                print(
                    f"replicate {replicate_index + 1}/{n_replicates}, "
                    f"temperature {temperature_index + 1}/{n_temperatures}: "
                    f"T={temperature:.6f}, seed={seed}",
                    flush=True,
                )

            replicate_runs.append(
                run_chain(
                    lattice_size=lattice_size,
                    temperature=float(temperature),
                    burn_in_sweeps=burn_in_sweeps,
                    production_sweeps=production_sweeps,
                    sample_every=sample_every,
                    seed=seed,
                    initial_state=initial_state,
                    coupling=coupling,
                    field=field,
                    boltzmann_constant=boltzmann_constant,
                    record_burn_in=record_burn_in,
                )
            )

        runs.append(replicate_runs)

    sweep: dict[str, object] = {
        "temperatures": temperature_grid.copy(),
        "seeds": seed_matrix.copy(),
        "n_replicates": int(n_replicates),
        "n_temperatures": int(n_temperatures),
        "lattice_size": int(lattice_size),
        "n_spins": int(lattice_size**2),
        "coupling": float(coupling),
        "field": float(field),
        "boltzmann_constant": float(boltzmann_constant),
        "burn_in_sweeps": int(burn_in_sweeps),
        "production_sweeps": int(production_sweeps),
        "sample_every": int(sample_every),
        "initial_state": str(initial_state),
        "record_burn_in": bool(record_burn_in),
        "elapsed_seconds": float(perf_counter() - start),
        "runs": runs,
    }

    fields = (
        "initial_lattice",
        "final_lattice",
        "burn_energy",
        "burn_magnetisation",
        "burn_acceptance",
        "production_sweep_numbers",
        "production_energy",
        "production_energy_squared",
        "production_magnetisation",
        "production_magnetisation_squared",
        "production_absolute_magnetisation",
        "production_acceptance",
        "mean_total_energy",
        "mean_energy_squared",
        "mean_energy_per_spin",
        "mean_signed_magnetisation",
        "mean_magnetisation_squared",
        "mean_absolute_magnetisation",
        "mean_acceptance_fraction",
    )

    for field_name in fields:
        sweep[field_name] = _stack_field(runs, field_name)

    return sweep


def split_half_means(values: Array) -> tuple[float, float]:
    """Return the means of the first and second halves of a trace."""
    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1 or samples.size < 2:
        raise ValueError("values must be one-dimensional with at least two entries.")

    midpoint = samples.size // 2
    return (
        float(np.mean(samples[:midpoint])),
        float(np.mean(samples[midpoint:])),
    )


def summary_rows(sweep: dict[str, object]) -> list[dict[str, object]]:
    """Convert a sweep result into one human-readable row per chain."""
    rows: list[dict[str, object]] = []
    n_spins = int(sweep["n_spins"])
    temperatures = np.asarray(sweep["temperatures"], dtype=float)
    seeds = np.asarray(sweep["seeds"], dtype=np.int64)
    energy = np.asarray(sweep["production_energy"], dtype=float)
    absolute_magnetisation = np.asarray(
        sweep["production_absolute_magnetisation"],
        dtype=float,
    )

    for replicate_index in range(int(sweep["n_replicates"])):
        for temperature_index, temperature in enumerate(temperatures):
            energy_per_spin_trace = (
                energy[replicate_index, temperature_index] / n_spins
            )
            absolute_trace = absolute_magnetisation[
                replicate_index,
                temperature_index,
            ]
            energy_first, energy_second = split_half_means(
                energy_per_spin_trace
            )
            magnetisation_first, magnetisation_second = split_half_means(
                absolute_trace
            )

            rows.append(
                {
                    "replicate": replicate_index,
                    "temperature": float(temperature),
                    "seed": int(seeds[replicate_index, temperature_index]),
                    "lattice_size": int(sweep["lattice_size"]),
                    "n_spins": n_spins,
                    "coupling": float(sweep["coupling"]),
                    "field": float(sweep["field"]),
                    "boltzmann_constant": float(
                        sweep["boltzmann_constant"]
                    ),
                    "initial_state": str(sweep["initial_state"]),
                    "burn_in_sweeps": int(sweep["burn_in_sweeps"]),
                    "production_sweeps": int(
                        sweep["production_sweeps"]
                    ),
                    "sample_every": int(sweep["sample_every"]),
                    "n_recorded_samples": int(energy_per_spin_trace.size),
                    "mean_total_energy": float(
                        np.asarray(sweep["mean_total_energy"])[
                            replicate_index,
                            temperature_index,
                        ]
                    ),
                    "mean_energy_squared": float(
                        np.asarray(sweep["mean_energy_squared"])[
                            replicate_index,
                            temperature_index,
                        ]
                    ),
                    "mean_energy_per_spin": float(
                        np.asarray(sweep["mean_energy_per_spin"])[
                            replicate_index,
                            temperature_index,
                        ]
                    ),
                    "mean_signed_magnetisation": float(
                        np.asarray(sweep["mean_signed_magnetisation"])[
                            replicate_index,
                            temperature_index,
                        ]
                    ),
                    "mean_magnetisation_squared": float(
                        np.asarray(sweep["mean_magnetisation_squared"])[
                            replicate_index,
                            temperature_index,
                        ]
                    ),
                    "mean_absolute_magnetisation": float(
                        np.asarray(sweep["mean_absolute_magnetisation"])[
                            replicate_index,
                            temperature_index,
                        ]
                    ),
                    "mean_acceptance_fraction": float(
                        np.asarray(sweep["mean_acceptance_fraction"])[
                            replicate_index,
                            temperature_index,
                        ]
                    ),
                    "first_half_mean_energy_per_spin": energy_first,
                    "second_half_mean_energy_per_spin": energy_second,
                    "first_half_mean_absolute_magnetisation": (
                        magnetisation_first
                    ),
                    "second_half_mean_absolute_magnetisation": (
                        magnetisation_second
                    ),
                }
            )

    return rows


def _serialisable_sweep_payload(sweep: dict[str, object]) -> dict[str, Array]:
    """Build an allow_pickle=False compatible NPZ payload."""
    scalar_fields = (
        "n_replicates",
        "n_temperatures",
        "lattice_size",
        "n_spins",
        "coupling",
        "field",
        "boltzmann_constant",
        "burn_in_sweeps",
        "production_sweeps",
        "sample_every",
        "record_burn_in",
        "elapsed_seconds",
    )
    array_fields = (
        "temperatures",
        "seeds",
        "initial_lattice",
        "final_lattice",
        "burn_energy",
        "burn_magnetisation",
        "burn_acceptance",
        "production_sweep_numbers",
        "production_energy",
        "production_energy_squared",
        "production_magnetisation",
        "production_magnetisation_squared",
        "production_absolute_magnetisation",
        "production_acceptance",
        "mean_total_energy",
        "mean_energy_squared",
        "mean_energy_per_spin",
        "mean_signed_magnetisation",
        "mean_magnetisation_squared",
        "mean_absolute_magnetisation",
        "mean_acceptance_fraction",
    )

    payload = {name: np.asarray(sweep[name]) for name in scalar_fields}
    payload.update({name: np.asarray(sweep[name]) for name in array_fields})
    payload["initial_state"] = np.asarray(str(sweep["initial_state"]))
    payload["exact_infinite_lattice_critical_temperature"] = np.asarray(
        exact_square_lattice_critical_temperature(
            coupling=float(sweep["coupling"]),
            boltzmann_constant=float(sweep["boltzmann_constant"]),
        )
    )
    payload["schema_version"] = np.asarray("1.0")
    return payload


def _file_sha256(path: Path) -> str:
    """Return digest for a configuration or result file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_temperature_sweep(
    sweep: dict[str, object],
    *,
    result_path: Path,
    summary_path: Path,
    metadata_path: Path,
    config: dict[str, Any],
    config_path: Path,
    overwrite: bool = False,
    additional_metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Save raw arrays, a summary CSV and reproducibility metadata."""
    paths = (result_path, summary_path, metadata_path)
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing output: {path}. "
                "Pass --overwrite to replace it."
            )

    np.savez_compressed(result_path, **_serialisable_sweep_payload(sweep))

    rows = summary_rows(sweep)
    fieldnames = list(rows[0].keys())
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metadata: dict[str, Any] = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": config.get("experiment", {}).get(
            "name",
            "unnamed_ising_experiment",
        ),
        "description": config.get("experiment", {}).get(
            "description",
            "",
        ),
        "configuration_file": str(config_path),
        "configuration_sha256": _file_sha256(config_path),
        "configuration": config,
        "result_file": str(result_path),
        "result_sha256": _file_sha256(result_path),
        "summary_file": str(summary_path),
        "physical_conventions": {
            "hamiltonian": "E = -J sum_<ij> s_i s_j - B sum_i s_i",
            "boundary_conditions": "periodic",
            "sweep_definition": (
                "Exactly L^2 random single-spin attempts with replacement."
            ),
            "magnetisation": "m = sum_i(s_i) / L^2",
            "energy_per_spin": "e = E / L^2",
            "specific_heat_per_spin": (
                "c_V = (<E^2> - <E>^2) / (N k_B T^2), using total E"
            ),
            "signed_susceptibility_per_spin": (
                "chi_signed = N (<m^2> - <m>^2) / (k_B T)"
            ),
            "absolute_centred_susceptibility_per_spin": (
                "chi_abs = N (<m^2> - <|m|>^2) / (k_B T)"
            ),
        },
        "runtime": {
            "elapsed_seconds": float(sweep["elapsed_seconds"]),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "seed_matrix": np.asarray(sweep["seeds"], dtype=np.int64).tolist(),
        "array_shapes": {
            name: list(np.asarray(value).shape)
            for name, value in _serialisable_sweep_payload(sweep).items()
        },
        "limitations": [
            f"Across-seed summaries use {int(sweep['n_replicates'])} independently seeded chain(s) per temperature.",
            "Uncertainty is provisional when the replicate count is small.",
            "Successive samples within each chain may be autocorrelated.",
            "The exact critical temperature is an infinite-lattice benchmark, not a fitted finite-L boundary.",
        ],
    }
    if additional_metadata:
        metadata.update(additional_metadata)

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "result": result_path,
        "summary": summary_path,
        "metadata": metadata_path,
    }


def save_chain_collection(
    chains: list[dict[str, object]],
    *,
    result_path: Path,
    summary_path: Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Save a small collection of diagnostic fixed-temperature chains."""
    for path in (result_path, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing output: {path}."
            )

    fields = (
        "initial_lattice",
        "final_lattice",
        "burn_energy",
        "burn_magnetisation",
        "burn_acceptance",
        "production_sweep_numbers",
        "production_energy",
        "production_energy_squared",
        "production_magnetisation",
        "production_magnetisation_squared",
        "production_absolute_magnetisation",
        "production_acceptance",
    )
    payload: dict[str, Array] = {
        "schema_version": np.asarray("1.0"),
        "temperatures": np.asarray(
            [chain["temperature"] for chain in chains],
            dtype=float,
        ),
        "seeds": np.asarray(
            [chain["seed"] for chain in chains],
            dtype=np.int64,
        ),
        "initial_states": np.asarray(
            [chain["initial_state"] for chain in chains]
        ),
    }
    for field_name in fields:
        payload[field_name] = np.stack(
            [np.asarray(chain[field_name]) for chain in chains],
            axis=0,
        )

    np.savez_compressed(result_path, **payload)

    summary_fields = [
        "temperature",
        "seed",
        "initial_state",
        "mean_energy_per_spin",
        "mean_signed_magnetisation",
        "mean_absolute_magnetisation",
        "mean_acceptance_fraction",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for chain in chains:
            writer.writerow({name: chain[name] for name in summary_fields})

    return {"result": result_path, "summary": summary_path}


def load_npz(path: Path | str) -> dict[str, Array]:
    """Load a project NPZ into a plain dictionary without pickle support."""
    with np.load(Path(path), allow_pickle=False) as data:
        return {name: data[name] for name in data.files}
