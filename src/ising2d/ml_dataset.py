"""Day 25 classical Ising configuration-dataset utilities."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .dataset import load_npz, make_seed_matrix, validate_temperature_grid
from .observables import (
    exact_square_lattice_critical_temperature,
    magnetisation,
    total_energy,
)
from .simulation import initialise_lattice, metropolis_sweep


Array = np.ndarray
DATASET_SCHEMA_VERSION = "1.0"
VALID_SPLIT_NAMES = {"train", "validation", "test"}
VALID_INITIAL_STATES = {"up", "down", "random"}
REQUIRED_OUTPUT_KEYS = {
    "dataset_npz",
    "manifest_csv",
    "summary_csv",
    "chain_diagnostics_csv",
    "figure_dir",
}

PER_SAMPLE_FIELDS = (
    "sample_ids",
    "temperatures",
    "temperature_indices",
    "replicate_indices",
    "chain_indices",
    "chain_ids",
    "seeds",
    "initial_states",
    "production_sweep_numbers",
    "sample_indices_within_chain",
    "split_names",
    "energy_per_spin",
    "signed_magnetisation",
    "absolute_magnetisation",
    "acceptance_fraction",
)

SERIALISABLE_FIELDS = (
    "schema_version",
    "lattice_size",
    "n_spins",
    "coupling",
    "field",
    "boltzmann_constant",
    "burn_in_sweeps",
    "production_sweeps",
    "configuration_interval",
    "configurations_per_chain",
    "n_replicates",
    "n_temperatures",
    "n_chains",
    "n_samples",
    "row_order",
    "flattening_order",
    "exact_infinite_lattice_critical_temperature",
    "configurations",
    *PER_SAMPLE_FIELDS,
    "seed_matrix",
    "initial_state_schedule",
    "split_schedule",
)


def _require_integer(value: Any, *, name: str, minimum: int) -> int:
    """Validate and return an integer-like configuration value."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not a Boolean value.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer.") from error
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be an integer.")
    integer = int(numeric)
    if integer < minimum:
        wording = "non-negative" if minimum == 0 else f"at least {minimum}"
        raise ValueError(f"{name} must be {wording}.")
    return integer


def validate_dataset_specification(
    specification: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the Day 25 YAML schema and return the parameters."""
    if not isinstance(specification, Mapping):
        raise ValueError("The YAML root must be a mapping.")

    missing_sections = {"schema_version", "experiment", "model", "dataset", "outputs"} - set(
        specification
    )
    if missing_sections:
        raise ValueError(
            "Configuration is missing keys or sections: "
            + ", ".join(sorted(missing_sections))
        )

    if str(specification["schema_version"]) != DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {DATASET_SCHEMA_VERSION!r}."
        )

    experiment = specification["experiment"]
    model = specification["model"]
    design = specification["dataset"]
    outputs = specification["outputs"]
    for name, section in (
        ("experiment", experiment),
        ("model", model),
        ("dataset", design),
        ("outputs", outputs),
    ):
        if not isinstance(section, Mapping):
            raise ValueError(f"{name} must be a mapping.")

    if not str(experiment.get("name", "")).strip():
        raise ValueError("experiment.name must be a non-empty string.")

    missing_outputs = REQUIRED_OUTPUT_KEYS - set(outputs)
    if missing_outputs:
        raise ValueError(
            "outputs is missing keys: " + ", ".join(sorted(missing_outputs))
        )

    temperatures = validate_temperature_grid(design["temperatures"])
    lattice_size = _require_integer(
        model["lattice_size"], name="model.lattice_size", minimum=2
    )
    coupling = float(model.get("coupling", 1.0))
    field = float(model.get("field", 0.0))
    boltzmann_constant = float(model.get("boltzmann_constant", 1.0))
    if not np.isfinite(coupling) or coupling <= 0.0:
        raise ValueError("model.coupling must be finite and positive.")
    if not np.isfinite(field) or field != 0.0:
        raise ValueError("Classical ML dataset v1 requires model.field=0.")
    if not np.isfinite(boltzmann_constant) or boltzmann_constant <= 0.0:
        raise ValueError("model.boltzmann_constant must be finite and positive.")

    burn_in_sweeps = _require_integer(
        design["burn_in_sweeps"], name="dataset.burn_in_sweeps", minimum=0
    )
    production_sweeps = _require_integer(
        design["production_sweeps"], name="dataset.production_sweeps", minimum=1
    )
    configuration_interval = _require_integer(
        design["configuration_interval"],
        name="dataset.configuration_interval",
        minimum=1,
    )
    configurations_per_chain = _require_integer(
        design["configurations_per_chain"],
        name="dataset.configurations_per_chain",
        minimum=1,
    )
    n_replicates = _require_integer(
        design["n_replicates"], name="dataset.n_replicates", minimum=1
    )
    base_seed = _require_integer(
        design["base_seed"], name="dataset.base_seed", minimum=0
    )

    if production_sweeps % configuration_interval != 0:
        raise ValueError(
            "dataset.configuration_interval must divide "
            "dataset.production_sweeps exactly."
        )
    expected_per_chain = production_sweeps // configuration_interval
    if configurations_per_chain != expected_per_chain:
        raise ValueError(
            "dataset.configurations_per_chain must equal "
            "production_sweeps // configuration_interval."
        )

    initial_state_schedule = [
        str(value) for value in design["initial_state_schedule"]
    ]
    split_schedule = [str(value) for value in design["split_schedule"]]
    if len(initial_state_schedule) != n_replicates:
        raise ValueError(
            "dataset.initial_state_schedule needs one entry per replicate."
        )
    if len(split_schedule) != n_replicates:
        raise ValueError("dataset.split_schedule needs one entry per replicate.")
    invalid_initial_states = sorted(
        set(initial_state_schedule) - VALID_INITIAL_STATES
    )
    if invalid_initial_states:
        raise ValueError(
            "Unsupported initial state(s): " + ", ".join(invalid_initial_states)
        )
    invalid_splits = sorted(set(split_schedule) - VALID_SPLIT_NAMES)
    if invalid_splits:
        raise ValueError("Unsupported split name(s): " + ", ".join(invalid_splits))

    seed_matrix = make_seed_matrix(base_seed, n_replicates, temperatures.size)
    if np.unique(seed_matrix).size != seed_matrix.size:
        raise ValueError("Every chain must have a unique seed.")

    return {
        "temperatures": temperatures,
        "lattice_size": lattice_size,
        "coupling": coupling,
        "field": field,
        "boltzmann_constant": boltzmann_constant,
        "burn_in_sweeps": burn_in_sweeps,
        "production_sweeps": production_sweeps,
        "configuration_interval": configuration_interval,
        "configurations_per_chain": configurations_per_chain,
        "n_replicates": n_replicates,
        "base_seed": base_seed,
        "initial_state_schedule": initial_state_schedule,
        "split_schedule": split_schedule,
        "seed_matrix": seed_matrix,
    }


def sample_configuration_chain(
    *,
    lattice_size: int,
    temperature: float,
    burn_in_sweeps: int,
    production_sweeps: int,
    configuration_interval: int,
    seed: int,
    initial_state: str,
    coupling: float = 1.0,
    field: float = 0.0,
    boltzmann_constant: float = 1.0,
) -> dict[str, Any]:
    """Run one chain and retain copied lattices at fixed production intervals."""
    lattice_size = _require_integer(
        lattice_size, name="lattice_size", minimum=2
    )
    burn_in_sweeps = _require_integer(
        burn_in_sweeps, name="burn_in_sweeps", minimum=0
    )
    production_sweeps = _require_integer(
        production_sweeps, name="production_sweeps", minimum=1
    )
    configuration_interval = _require_integer(
        configuration_interval, name="configuration_interval", minimum=1
    )
    seed = _require_integer(seed, name="seed", minimum=0)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive.")
    if production_sweeps % configuration_interval != 0:
        raise ValueError(
            "configuration_interval must divide production_sweeps exactly."
        )
    if initial_state not in VALID_INITIAL_STATES:
        raise ValueError("initial_state must be 'up', 'down' or 'random'.")

    rng = np.random.default_rng(seed)
    lattice = initialise_lattice(lattice_size, rng, mode=initial_state)

    for _ in range(burn_in_sweeps):
        metropolis_sweep(
            lattice,
            temperature,
            rng,
            coupling=coupling,
            field=field,
            boltzmann_constant=boltzmann_constant,
        )

    n_saved = production_sweeps // configuration_interval
    configurations = np.empty(
        (n_saved, lattice_size, lattice_size), dtype=np.int8
    )
    production_sweep_numbers = np.empty(n_saved, dtype=np.int32)
    energy_per_spin = np.empty(n_saved, dtype=np.float64)
    signed_magnetisation = np.empty(n_saved, dtype=np.float64)
    absolute_magnetisation = np.empty(n_saved, dtype=np.float64)
    acceptance_fraction = np.empty(n_saved, dtype=np.float64)

    accepted_fraction_sum = 0.0
    saved_index = 0
    n_spins = lattice_size**2

    for production_sweep in range(1, production_sweeps + 1):
        accepted_fraction_sum += metropolis_sweep(
            lattice,
            temperature,
            rng,
            coupling=coupling,
            field=field,
            boltzmann_constant=boltzmann_constant,
        )
        if production_sweep % configuration_interval == 0:
            configurations[saved_index] = lattice.copy()
            production_sweep_numbers[saved_index] = production_sweep
            energy_per_spin[saved_index] = (
                total_energy(lattice, coupling=coupling, field=field) / n_spins
            )
            signed_magnetisation[saved_index] = magnetisation(lattice)
            absolute_magnetisation[saved_index] = abs(
                signed_magnetisation[saved_index]
            )
            acceptance_fraction[saved_index] = (
                accepted_fraction_sum / configuration_interval
            )
            accepted_fraction_sum = 0.0
            saved_index += 1

    return {
        "configurations": configurations,
        "production_sweep_numbers": production_sweep_numbers,
        "sample_indices_within_chain": np.arange(n_saved, dtype=np.int16),
        "energy_per_spin": energy_per_spin,
        "signed_magnetisation": signed_magnetisation,
        "absolute_magnetisation": absolute_magnetisation,
        "acceptance_fraction": acceptance_fraction,
        "final_lattice": lattice.copy(),
    }


def build_classical_ml_dataset(
    specification: Mapping[str, Any],
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """Build a deterministic temperature-major configuration dataset."""
    parameters = validate_dataset_specification(specification)
    temperatures = parameters["temperatures"]
    lattice_size = parameters["lattice_size"]
    coupling = parameters["coupling"]
    field = parameters["field"]
    boltzmann_constant = parameters["boltzmann_constant"]
    burn_in_sweeps = parameters["burn_in_sweeps"]
    production_sweeps = parameters["production_sweeps"]
    configuration_interval = parameters["configuration_interval"]
    configurations_per_chain = parameters["configurations_per_chain"]
    n_replicates = parameters["n_replicates"]
    initial_state_schedule = parameters["initial_state_schedule"]
    split_schedule = parameters["split_schedule"]
    seed_matrix = parameters["seed_matrix"]

    n_temperatures = int(temperatures.size)
    n_chains = n_temperatures * n_replicates
    n_samples = n_chains * configurations_per_chain
    exact_tc = exact_square_lattice_critical_temperature(
        coupling=coupling,
        boltzmann_constant=boltzmann_constant,
    )

    configurations = np.empty(
        (n_samples, lattice_size, lattice_size), dtype=np.int8
    )
    sample_ids = np.empty(n_samples, dtype="<U24")
    sample_temperatures = np.empty(n_samples, dtype=np.float64)
    temperature_indices = np.empty(n_samples, dtype=np.int16)
    replicate_indices = np.empty(n_samples, dtype=np.int16)
    chain_indices = np.empty(n_samples, dtype=np.int32)
    chain_ids = np.empty(n_samples, dtype="<U40")
    seeds = np.empty(n_samples, dtype=np.int64)
    initial_states = np.empty(n_samples, dtype="<U6")
    production_sweep_numbers = np.empty(n_samples, dtype=np.int32)
    sample_indices_within_chain = np.empty(n_samples, dtype=np.int16)
    split_names = np.empty(n_samples, dtype="<U10")
    energy_per_spin = np.empty(n_samples, dtype=np.float64)
    signed_magnetisation = np.empty(n_samples, dtype=np.float64)
    absolute_magnetisation = np.empty(n_samples, dtype=np.float64)
    acceptance_fraction = np.empty(n_samples, dtype=np.float64)

    started = perf_counter()
    row_start = 0
    chain_index = 0

    for temperature_index, temperature in enumerate(temperatures):
        for replicate_index in range(n_replicates):
            seed = int(seed_matrix[replicate_index, temperature_index])
            initial_state = initial_state_schedule[replicate_index]
            split_name = split_schedule[replicate_index]
            chain_id = (
                f"ising2d_t{temperature_index:02d}_r{replicate_index:02d}_seed{seed}"
            )

            if verbose:
                print(
                    f"chain {chain_index + 1:03d}/{n_chains}: "
                    f"T={temperature:.9f}, replicate={replicate_index}, "
                    f"start={initial_state}, split={split_name}, seed={seed}",
                    flush=True,
                )

            chain = sample_configuration_chain(
                lattice_size=lattice_size,
                temperature=float(temperature),
                burn_in_sweeps=burn_in_sweeps,
                production_sweeps=production_sweeps,
                configuration_interval=configuration_interval,
                seed=seed,
                initial_state=initial_state,
                coupling=coupling,
                field=field,
                boltzmann_constant=boltzmann_constant,
            )

            row_stop = row_start + configurations_per_chain
            rows = slice(row_start, row_stop)
            local_indices = np.arange(
                configurations_per_chain, dtype=np.int16
            )

            configurations[rows] = chain["configurations"]
            sample_ids[rows] = [
                f"d25_t{temperature_index:02d}_r{replicate_index:02d}_s{sample_index:03d}"
                for sample_index in local_indices
            ]
            sample_temperatures[rows] = float(temperature)
            temperature_indices[rows] = temperature_index
            replicate_indices[rows] = replicate_index
            chain_indices[rows] = chain_index
            chain_ids[rows] = chain_id
            seeds[rows] = seed
            initial_states[rows] = initial_state
            production_sweep_numbers[rows] = chain[
                "production_sweep_numbers"
            ]
            sample_indices_within_chain[rows] = local_indices
            split_names[rows] = split_name
            energy_per_spin[rows] = chain["energy_per_spin"]
            signed_magnetisation[rows] = chain["signed_magnetisation"]
            absolute_magnetisation[rows] = chain["absolute_magnetisation"]
            acceptance_fraction[rows] = chain["acceptance_fraction"]

            row_start = row_stop
            chain_index += 1

    return {
        "schema_version": np.asarray(DATASET_SCHEMA_VERSION),
        "lattice_size": np.asarray(lattice_size, dtype=np.int32),
        "n_spins": np.asarray(lattice_size**2, dtype=np.int32),
        "coupling": np.asarray(coupling, dtype=np.float64),
        "field": np.asarray(field, dtype=np.float64),
        "boltzmann_constant": np.asarray(
            boltzmann_constant, dtype=np.float64
        ),
        "burn_in_sweeps": np.asarray(burn_in_sweeps, dtype=np.int32),
        "production_sweeps": np.asarray(
            production_sweeps, dtype=np.int32
        ),
        "configuration_interval": np.asarray(
            configuration_interval, dtype=np.int32
        ),
        "configurations_per_chain": np.asarray(
            configurations_per_chain, dtype=np.int32
        ),
        "n_replicates": np.asarray(n_replicates, dtype=np.int16),
        "n_temperatures": np.asarray(n_temperatures, dtype=np.int16),
        "n_chains": np.asarray(n_chains, dtype=np.int32),
        "n_samples": np.asarray(n_samples, dtype=np.int32),
        "row_order": np.asarray(
            "temperature_index,replicate_index,sample_index_within_chain"
        ),
        "flattening_order": np.asarray("NumPy C order (row-major)"),
        "exact_infinite_lattice_critical_temperature": np.asarray(
            exact_tc, dtype=np.float64
        ),
        "configurations": configurations,
        "sample_ids": sample_ids,
        "temperatures": sample_temperatures,
        "temperature_indices": temperature_indices,
        "replicate_indices": replicate_indices,
        "chain_indices": chain_indices,
        "chain_ids": chain_ids,
        "seeds": seeds,
        "initial_states": initial_states,
        "production_sweep_numbers": production_sweep_numbers,
        "sample_indices_within_chain": sample_indices_within_chain,
        "split_names": split_names,
        "energy_per_spin": energy_per_spin,
        "signed_magnetisation": signed_magnetisation,
        "absolute_magnetisation": absolute_magnetisation,
        "acceptance_fraction": acceptance_fraction,
        "seed_matrix": seed_matrix.astype(np.int64),
        "initial_state_schedule": np.asarray(
            initial_state_schedule, dtype="<U6"
        ),
        "split_schedule": np.asarray(split_schedule, dtype="<U10"),
        "elapsed_seconds": float(perf_counter() - started),
    }


def serialisable_payload(
    dataset: Mapping[str, Any],
) -> dict[str, Array]:
    """Return the non-object arrays written to the canonical NPZ."""
    missing = [name for name in SERIALISABLE_FIELDS if name not in dataset]
    if missing:
        raise ValueError(
            "Dataset is missing serialisable fields: " + ", ".join(missing)
        )
    payload = {
        name: np.asarray(dataset[name]) for name in SERIALISABLE_FIELDS
    }
    object_fields = [
        name for name, value in payload.items() if value.dtype == object
    ]
    if object_fields:
        raise ValueError(f"Object arrays are not allowed: {object_fields}")
    return payload


def load_previous_day_seeds(raw_directory: Path | str) -> Array:
    """Collect saved Day 22 and Day 23 seeds for the non-overlap check."""
    directory = Path(raw_directory)
    arrays: list[Array] = []
    for path in sorted(directory.glob("ising_day2[23]*.npz")):
        with np.load(path, allow_pickle=False) as saved:
            if "seeds" in saved.files:
                arrays.append(
                    np.asarray(saved["seeds"], dtype=np.int64).ravel()
                )
    if not arrays:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(arrays))


def manifest_from_dataset(dataset: Mapping[str, Any]) -> pd.DataFrame:
    """Create the row-aligned provenance manifest."""
    return pd.DataFrame(
        {
            "row_index": np.arange(
                int(np.asarray(dataset["n_samples"]).item()), dtype=np.int32
            ),
            "sample_id": dataset["sample_ids"],
            "temperature": dataset["temperatures"],
            "temperature_index": dataset["temperature_indices"],
            "replicate_index": dataset["replicate_indices"],
            "chain_index": dataset["chain_indices"],
            "chain_id": dataset["chain_ids"],
            "seed": dataset["seeds"],
            "initial_state": dataset["initial_states"],
            "production_sweep_number": dataset[
                "production_sweep_numbers"
            ],
            "sample_index_within_chain": dataset[
                "sample_indices_within_chain"
            ],
            "split_name": dataset["split_names"],
            "energy_per_spin": dataset["energy_per_spin"],
            "signed_magnetisation": dataset["signed_magnetisation"],
            "absolute_magnetisation": dataset[
                "absolute_magnetisation"
            ],
            "acceptance_fraction": dataset["acceptance_fraction"],
        }
    )


def make_summary_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """Summarise counts and basic observables by temperature, split and start."""
    return (
        manifest.groupby(
            [
                "temperature_index",
                "temperature",
                "split_name",
                "initial_state",
            ],
            as_index=False,
        )
        .agg(
            configuration_count=("sample_id", "size"),
            chain_count=("chain_id", "nunique"),
            mean_energy_per_spin=("energy_per_spin", "mean"),
            mean_signed_magnetisation=("signed_magnetisation", "mean"),
            mean_absolute_magnetisation=(
                "absolute_magnetisation",
                "mean",
            ),
            mean_acceptance_fraction=("acceptance_fraction", "mean"),
        )
        .sort_values(
            ["temperature_index", "split_name", "initial_state"]
        )
        .reset_index(drop=True)
    )


def _lag_one_correlation(values: Array) -> float:
    values = np.asarray(values, dtype=float)
    if (
        values.size < 3
        or np.std(values[:-1]) == 0.0
        or np.std(values[1:]) == 0.0
    ):
        return float("nan")
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def chain_spacing_diagnostics(
    dataset: Mapping[str, Any],
) -> pd.DataFrame:
    """Report retained-state spacing without deleting physical repetitions."""
    configurations = np.asarray(dataset["configurations"], dtype=np.int8)
    manifest = manifest_from_dataset(dataset)
    rows: list[dict[str, Any]] = []

    for chain_index, chain_rows in manifest.groupby(
        "chain_index", sort=True
    ):
        chain_rows = chain_rows.sort_values("sample_index_within_chain")
        indices = chain_rows["row_index"].to_numpy(dtype=int)
        flattened = configurations[indices].reshape(indices.size, -1)
        if flattened.shape[0] > 1:
            consecutive_hamming = np.mean(
                flattened[1:] != flattened[:-1], axis=1
            )
            consecutive_duplicates = np.all(
                flattened[1:] == flattened[:-1], axis=1
            )
            mean_hamming = float(np.mean(consecutive_hamming))
            minimum_hamming = float(np.min(consecutive_hamming))
            duplicate_rate = float(np.mean(consecutive_duplicates))
        else:
            mean_hamming = float("nan")
            minimum_hamming = float("nan")
            duplicate_rate = float("nan")
        unique_fraction = (
            np.unique(flattened, axis=0).shape[0] / flattened.shape[0]
        )

        first = chain_rows.iloc[0]
        rows.append(
            {
                "chain_index": int(chain_index),
                "chain_id": str(first["chain_id"]),
                "temperature_index": int(first["temperature_index"]),
                "temperature": float(first["temperature"]),
                "replicate_index": int(first["replicate_index"]),
                "seed": int(first["seed"]),
                "initial_state": str(first["initial_state"]),
                "split_name": str(first["split_name"]),
                "configuration_count": int(indices.size),
                "mean_consecutive_hamming_fraction": mean_hamming,
                "minimum_consecutive_hamming_fraction": minimum_hamming,
                "consecutive_duplicate_rate": duplicate_rate,
                "all_rows_duplicate_rate": float(1.0 - unique_fraction),
                "signed_magnetisation_lag1": _lag_one_correlation(
                    chain_rows["signed_magnetisation"].to_numpy()
                ),
                "absolute_magnetisation_lag1": _lag_one_correlation(
                    chain_rows["absolute_magnetisation"].to_numpy()
                ),
            }
        )

    return pd.DataFrame(rows)


def validate_dataset(
    dataset: Mapping[str, Any],
    specification: Mapping[str, Any],
    previous_seeds: Array | None = None,
) -> dict[str, Any]:
    """Validate configuration, metadata, split and observable integrity."""
    validate_dataset_specification(specification)
    payload = serialisable_payload(dataset)
    configurations = payload["configurations"]
    n_samples = int(payload["n_samples"].item())
    lattice_size = int(payload["lattice_size"].item())
    n_temperatures = int(payload["n_temperatures"].item())
    n_replicates = int(payload["n_replicates"].item())
    configurations_per_chain = int(
        payload["configurations_per_chain"].item()
    )
    interval = int(payload["configuration_interval"].item())

    expected_shape = (n_samples, lattice_size, lattice_size)
    if configurations.shape != expected_shape:
        raise ValueError(
            f"Expected configurations.shape={expected_shape}, "
            f"got {configurations.shape}."
        )
    if configurations.dtype != np.int8:
        raise ValueError("configurations must have dtype int8.")
    if not np.all(np.isin(configurations, (-1, 1))):
        raise ValueError("configurations must contain only -1 and +1.")

    for name in PER_SAMPLE_FIELDS:
        values = payload[name]
        if values.ndim != 1 or values.shape[0] != n_samples:
            raise ValueError(f"{name} is not aligned with configurations.")
    if np.unique(payload["sample_ids"]).size != n_samples:
        raise ValueError("sample_ids are not unique.")

    expected_samples = (
        n_temperatures * n_replicates * configurations_per_chain
    )
    if n_samples != expected_samples:
        raise ValueError("n_samples disagrees with the configured design.")

    chain_indices = payload["chain_indices"].astype(int)
    chain_ids = payload["chain_ids"].astype(str)
    seeds = payload["seeds"].astype(np.int64)
    split_names = payload["split_names"].astype(str)
    temperature_indices = payload["temperature_indices"].astype(int)
    sample_indices = payload["sample_indices_within_chain"].astype(int)
    sweep_numbers = payload["production_sweep_numbers"].astype(int)

    for chain_index in np.unique(chain_indices):
        mask = chain_indices == chain_index
        order = np.argsort(sample_indices[mask])
        if int(np.sum(mask)) != configurations_per_chain:
            raise ValueError(
                f"Chain {chain_index} has the wrong number of configurations."
            )
        for values, name in (
            (chain_ids[mask], "chain_id"),
            (seeds[mask], "seed"),
            (split_names[mask], "split_name"),
            (temperature_indices[mask], "temperature_index"),
        ):
            if np.unique(values).size != 1:
                raise ValueError(f"{name} changes inside chain {chain_index}.")
        if not np.array_equal(
            sample_indices[mask][order],
            np.arange(configurations_per_chain),
        ):
            raise ValueError(
                f"Chain {chain_index} has incomplete sample indices."
            )
        expected_sweeps = interval * np.arange(
            1, configurations_per_chain + 1
        )
        if not np.array_equal(sweep_numbers[mask][order], expected_sweeps):
            raise ValueError(
                f"Chain {chain_index} has incorrect retained sweep numbers."
            )

    chain_split_pairs = pd.DataFrame(
        {"chain_id": chain_ids, "seed": seeds, "split_name": split_names}
    ).drop_duplicates()
    if chain_split_pairs.groupby("chain_id")["split_name"].nunique().max() != 1:
        raise ValueError("A chain appears in more than one split.")
    if chain_split_pairs.groupby("seed")["split_name"].nunique().max() != 1:
        raise ValueError("A seed appears in more than one split.")
    if chain_split_pairs["seed"].nunique() != int(payload["n_chains"].item()):
        raise ValueError("Each chain must have a unique seed.")

    for split_name in np.unique(split_names):
        represented = np.unique(
            temperature_indices[split_names == split_name]
        )
        if not np.array_equal(represented, np.arange(n_temperatures)):
            raise ValueError(
                f"Split {split_name} does not contain the complete temperature grid."
            )

    if previous_seeds is not None:
        previous = np.asarray(previous_seeds, dtype=np.int64)
        if previous.size and np.intersect1d(
            np.unique(seeds), previous
        ).size:
            raise ValueError(
                "Day 25 seeds overlap recorded Day 22 or Day 23 seeds."
            )

    flattened = configurations.reshape(
        n_samples, lattice_size**2, order="C"
    )
    if not np.array_equal(
        flattened.reshape(configurations.shape, order="C"), configurations
    ):
        raise ValueError("C-order flattening and reshaping did not round-trip.")

    coupling = float(payload["coupling"].item())
    field = float(payload["field"].item())
    recomputed_energy = np.asarray(
        [
            total_energy(
                configuration, coupling=coupling, field=field
            )
            / configuration.size
            for configuration in configurations
        ]
    )
    recomputed_magnetisation = np.asarray(
        [magnetisation(configuration) for configuration in configurations]
    )
    if not np.allclose(
        recomputed_energy,
        payload["energy_per_spin"],
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(
            "Stored energies disagree with recomputation from configurations."
        )
    if not np.allclose(
        recomputed_magnetisation,
        payload["signed_magnetisation"],
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(
            "Stored signed magnetisations disagree with recomputation."
        )
    if not np.allclose(
        np.abs(recomputed_magnetisation),
        payload["absolute_magnetisation"],
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(
            "Stored absolute magnetisations disagree with recomputation."
        )

    test_configuration = configurations[0]
    if not np.isclose(
        total_energy(test_configuration, coupling=coupling, field=0.0),
        total_energy(-test_configuration, coupling=coupling, field=0.0),
    ):
        raise ValueError("Global spin flip did not preserve energy at B=0.")
    if not np.isclose(
        magnetisation(-test_configuration),
        -magnetisation(test_configuration),
    ):
        raise ValueError("Global spin flip did not reverse magnetisation.")

    temperatures = payload["temperatures"].astype(float)
    signed_m = payload["signed_magnetisation"].astype(float)
    absolute_m = payload["absolute_magnetisation"].astype(float)
    energy = payload["energy_per_spin"].astype(float)
    initial_states = payload["initial_states"].astype(str)
    low_temperature = float(np.min(temperatures))
    high_temperature = float(np.max(temperatures))
    low_mask = temperatures == low_temperature
    high_mask = temperatures == high_temperature
    low_signs = set(np.sign(signed_m[low_mask]).astype(int))
    if not {-1, 1}.issubset(low_signs):
        raise ValueError(
            "Both low-temperature magnetisation sectors are not represented."
        )

    for split_name in np.unique(split_names):
        split_mask = low_mask & (split_names == split_name)
        split_low_signs = set(
            np.sign(signed_m[split_mask]).astype(int)
        )
        configured_starts = set(initial_states[split_mask])
        if {"up", "down"}.issubset(configured_starts) and not {
            -1,
            1,
        }.issubset(split_low_signs):
            raise ValueError(
                f"Split {split_name} does not retain both low-temperature sectors."
            )

    diagnostics = chain_spacing_diagnostics(dataset)
    return {
        "all_integrity_checks_passed": True,
        "configuration_shape": list(configurations.shape),
        "configuration_dtype": str(configurations.dtype),
        "n_samples": n_samples,
        "n_chains": int(payload["n_chains"].item()),
        "unique_seed_count": int(np.unique(seeds).size),
        "low_temperature_mean_absolute_magnetisation": float(
            np.mean(absolute_m[low_mask])
        ),
        "high_temperature_mean_absolute_magnetisation": float(
            np.mean(absolute_m[high_mask])
        ),
        "low_temperature_mean_energy_per_spin": float(
            np.mean(energy[low_mask])
        ),
        "high_temperature_mean_energy_per_spin": float(
            np.mean(energy[high_mask])
        ),
        "both_low_temperature_sectors": True,
        "mean_consecutive_hamming_fraction": float(
            diagnostics["mean_consecutive_hamming_fraction"].mean()
        ),
        "mean_consecutive_duplicate_rate": float(
            diagnostics["consecutive_duplicate_rate"].mean()
        ),
    }


def resolve_outputs(
    specification: Mapping[str, Any],
    *,
    project_root: Path | str,
) -> dict[str, Path]:
    """Resolve the five lean output paths relative to the project root."""
    root = Path(project_root)
    paths: dict[str, Path] = {}
    for key in REQUIRED_OUTPUT_KEYS:
        value = Path(str(specification["outputs"][key]))
        paths[key] = value if value.is_absolute() else root / value
    return paths


def expected_figure_paths(
    specification: Mapping[str, Any],
    *,
    project_root: Path | str,
) -> dict[str, Path]:
    """Return the four validation-figure destinations."""
    paths = resolve_outputs(specification, project_root=project_root)
    prefix = str(specification["experiment"]["name"])
    figure_dir = paths["figure_dir"]
    return {
        "montage": figure_dir / f"{prefix}_configuration_montage.png",
        "counts": figure_dir
        / f"{prefix}_counts_by_temperature_and_split.png",
        "physics": figure_dir / f"{prefix}_dataset_physics_check.png",
        "spacing": figure_dir
        / f"{prefix}_configuration_spacing_diagnostic.png",
    }


def require_output_paths_available(
    specification: Mapping[str, Any],
    *,
    project_root: Path | str,
    overwrite: bool,
) -> dict[str, Path]:
    """Refuse accidental replacement of data, tables or figures."""
    paths = resolve_outputs(specification, project_root=project_root)
    if overwrite:
        return paths

    existing = [
        path
        for key, path in paths.items()
        if key != "figure_dir" and path.exists()
    ]
    existing.extend(
        path
        for path in expected_figure_paths(
            specification, project_root=project_root
        ).values()
        if path.exists()
    )
    if existing:
        joined = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Refusing to replace existing Day 25 outputs:\n" + joined
        )
    return paths


def save_dataset(
    specification: Mapping[str, Any],
    dataset: Mapping[str, Any],
    *,
    project_root: Path | str,
    overwrite: bool,
) -> dict[str, Any]:
    """Save the lean NPZ, manifest, summary and chain diagnostics."""
    paths = require_output_paths_available(
        specification,
        project_root=project_root,
        overwrite=overwrite,
    )
    for key, path in paths.items():
        if key == "figure_dir":
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)

    payload = serialisable_payload(dataset)
    manifest = manifest_from_dataset(dataset)
    summary = make_summary_table(manifest)
    diagnostics = chain_spacing_diagnostics(dataset)

    np.savez_compressed(paths["dataset_npz"], **payload)
    manifest.to_csv(paths["manifest_csv"], index=False)
    summary.to_csv(paths["summary_csv"], index=False)
    diagnostics.to_csv(paths["chain_diagnostics_csv"], index=False)

    reloaded = load_npz(paths["dataset_npz"])
    if set(reloaded) != set(payload):
        raise ValueError("Reloaded NPZ fields differ from the saved payload.")
    for name, values in payload.items():
        if not np.array_equal(reloaded[name], values):
            raise ValueError(
                f"Reloaded field {name} differs from the in-memory array."
            )

    saved_manifest = pd.read_csv(paths["manifest_csv"])
    if not np.array_equal(
        saved_manifest["sample_id"].astype(str).to_numpy(),
        np.asarray(dataset["sample_ids"]).astype(str),
    ):
        raise ValueError("Manifest rows are not aligned with NPZ rows.")

    return {
        "paths": paths,
        "manifest": manifest,
        "summary": summary,
        "diagnostics": diagnostics,
        "reloaded": reloaded,
    }


def create_validation_figures(
    specification: Mapping[str, Any],
    dataset: Mapping[str, Any],
    diagnostics: pd.DataFrame,
    *,
    project_root: Path | str,
    overwrite: bool,
    day23_summary_path: Path | str | None = None,
) -> tuple[dict[str, Path], pd.DataFrame]:
    """Create the four Day 25 validation figures."""
    figure_paths = expected_figure_paths(
        specification, project_root=project_root
    )
    for path in figure_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        existing = [path for path in figure_paths.values() if path.exists()]
        if existing:
            raise FileExistsError(
                f"Refusing to overwrite validation figures: {existing}"
            )

    manifest = manifest_from_dataset(dataset)
    configurations = np.asarray(dataset["configurations"])
    exact_tc = float(
        np.asarray(
            dataset["exact_infinite_lattice_critical_temperature"]
        ).item()
    )
    low_temperature = float(manifest["temperature"].min())
    high_temperature = float(manifest["temperature"].max())
    near_temperature = float(
        manifest.loc[
            (manifest["temperature"] - exact_tc).abs().idxmin(),
            "temperature",
        ]
    )

    def row_for(mask: pd.Series, *, chain_number: int = 0) -> pd.Series:
        candidates = manifest.loc[mask]
        chain_id = candidates["chain_id"].drop_duplicates().iloc[
            chain_number
        ]
        return candidates.loc[candidates["chain_id"] == chain_id].iloc[-1]

    montage_rows = [
        manifest.loc[
            (manifest["temperature"] == low_temperature)
            & (manifest["signed_magnetisation"] > 0)
        ]
        .sort_values("signed_magnetisation")
        .iloc[-1],
        manifest.loc[
            (manifest["temperature"] == low_temperature)
            & (manifest["signed_magnetisation"] < 0)
        ]
        .sort_values("signed_magnetisation")
        .iloc[0],
        row_for(manifest["temperature"] == near_temperature, chain_number=0),
        row_for(manifest["temperature"] == near_temperature, chain_number=1),
        row_for(manifest["temperature"] == high_temperature, chain_number=0),
        row_for(manifest["temperature"] == high_temperature, chain_number=1),
    ]

    figure, axes = plt.subplots(2, 3, figsize=(12, 8))
    for axis, row in zip(axes.ravel(), montage_rows, strict=True):
        index = int(row["row_index"])
        axis.imshow(
            configurations[index],
            cmap="binary",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
        )
        axis.set_title(
            f"T={row['temperature']:.6f}; chain={int(row['chain_index'])}; "
            f"seed={int(row['seed'])}\n"
            f"start={row['initial_state']}; "
            f"m={row['signed_magnetisation']:+.3f}; "
            f"sweep={int(row['production_sweep_number'])}",
            fontsize=8,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("Classical Ising configuration validation montage")
    figure.tight_layout()
    figure.savefig(figure_paths["montage"], dpi=250, bbox_inches="tight")
    plt.close(figure)

    counts = (
        manifest.groupby(["temperature", "split_name"], as_index=False)
        .size()
        .rename(columns={"size": "configuration_count"})
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    for split_name, rows in counts.groupby("split_name"):
        axis.plot(
            rows["temperature"],
            rows["configuration_count"],
            marker="o",
            label=split_name,
        )
    axis.set_xlabel(r"Temperature, $k_B T/J$")
    axis.set_ylabel("Saved configurations")
    axis.set_title(
        "Dataset count by temperature and complete-chain split"
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_paths["counts"], dpi=250, bbox_inches="tight")
    plt.close(figure)

    day25_physics = (
        manifest.groupby("temperature", as_index=False)
        .agg(
            day25_energy_per_spin=("energy_per_spin", "mean"),
            day25_absolute_magnetisation=(
                "absolute_magnetisation",
                "mean",
            ),
        )
        .sort_values("temperature")
    )
    comparison = day25_physics.copy()
    if day23_summary_path is not None and Path(day23_summary_path).exists():
        day23 = pd.read_csv(day23_summary_path)[
            [
                "temperature",
                "mean_energy_per_spin_mean",
                "mean_absolute_magnetisation_mean",
            ]
        ].rename(
            columns={
                "mean_energy_per_spin_mean": "day23_energy_per_spin",
                "mean_absolute_magnetisation_mean": (
                    "day23_absolute_magnetisation"
                ),
            }
        )
        comparison = day25_physics.merge(
            day23, on="temperature", how="left"
        )
        comparison["energy_difference_day25_minus_day23"] = (
            comparison["day25_energy_per_spin"]
            - comparison["day23_energy_per_spin"]
        )
        comparison[
            "absolute_magnetisation_difference_day25_minus_day23"
        ] = (
            comparison["day25_absolute_magnetisation"]
            - comparison["day23_absolute_magnetisation"]
        )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(
        comparison["temperature"],
        comparison["day25_energy_per_spin"],
        marker="o",
        label="Day 25 saved configurations",
    )
    axes[1].plot(
        comparison["temperature"],
        comparison["day25_absolute_magnetisation"],
        marker="o",
        label="Day 25 saved configurations",
    )
    if "day23_energy_per_spin" in comparison:
        axes[0].plot(
            comparison["temperature"],
            comparison["day23_energy_per_spin"],
            marker="x",
            linestyle="--",
            label="Day 23 baseline",
        )
        axes[1].plot(
            comparison["temperature"],
            comparison["day23_absolute_magnetisation"],
            marker="x",
            linestyle="--",
            label="Day 23 baseline",
        )
    for axis in axes:
        axis.axvline(
            exact_tc,
            linestyle=":",
            linewidth=1.0,
            label="exact infinite-L benchmark",
        )
        axis.set_xlabel(r"Temperature, $k_B T/J$")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Mean energy per spin")
    axes[0].set_title("Energy from saved configurations")
    axes[1].set_ylabel("Mean absolute magnetisation")
    axes[1].set_title("Magnetic order from saved configurations")
    figure.tight_layout()
    figure.savefig(figure_paths["physics"], dpi=250, bbox_inches="tight")
    plt.close(figure)

    spacing = (
        diagnostics.groupby("temperature", as_index=False)
        .agg(
            mean_consecutive_hamming_fraction=(
                "mean_consecutive_hamming_fraction",
                "mean",
            ),
            mean_consecutive_duplicate_rate=(
                "consecutive_duplicate_rate",
                "mean",
            ),
            mean_signed_magnetisation_lag1=(
                "signed_magnetisation_lag1",
                "mean",
            ),
        )
        .sort_values("temperature")
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    axes[0].plot(
        spacing["temperature"],
        spacing["mean_consecutive_hamming_fraction"],
        marker="o",
    )
    axes[1].plot(
        spacing["temperature"],
        spacing["mean_consecutive_duplicate_rate"],
        marker="o",
    )
    axes[2].plot(
        spacing["temperature"],
        spacing["mean_signed_magnetisation_lag1"],
        marker="o",
    )
    axes[0].set_ylabel("Mean consecutive Hamming fraction")
    axes[1].set_ylabel("Consecutive exact-duplicate proportion")
    axes[2].set_ylabel("Retained signed-m lag-one correlation")
    for axis in axes:
        axis.axvline(exact_tc, linestyle=":", linewidth=1.0)
        axis.set_xlabel(r"Temperature, $k_B T/J$")
        axis.grid(True, alpha=0.3)
    figure.suptitle(
        "Configuration-spacing diagnostics "
        "(Hamming distance is not proof of independence)"
    )
    figure.tight_layout()
    figure.savefig(figure_paths["spacing"], dpi=250, bbox_inches="tight")
    plt.close(figure)

    return figure_paths, comparison
