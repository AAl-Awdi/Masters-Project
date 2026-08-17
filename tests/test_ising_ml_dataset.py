from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ising2d.ml_dataset import (
    build_classical_ml_dataset,
    sample_configuration_chain,
    save_dataset,
    serialisable_payload,
    validate_dataset,
    validate_dataset_specification,
)
from ising2d.observables import magnetisation, total_energy


def tiny_specification() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "experiment": {
            "name": "day25_test_dataset",
            "description": "Small deterministic test dataset.",
        },
        "model": {
            "lattice_size": 4,
            "coupling": 1.0,
            "field": 0.0,
            "boltzmann_constant": 1.0,
        },
        "dataset": {
            "temperatures": [1.5, 3.5],
            "burn_in_sweeps": 2,
            "production_sweeps": 4,
            "configuration_interval": 2,
            "configurations_per_chain": 2,
            "n_replicates": 2,
            "base_seed": 25900,
            "initial_state_schedule": ["up", "down"],
            "split_schedule": ["train", "test"],
        },
        "outputs": {
            "dataset_npz": "data/raw/classical/test_dataset.npz",
            "manifest_csv": "data/processed/classical/test_manifest.csv",
            "summary_csv": "data/processed/classical/test_summary.csv",
            "chain_diagnostics_csv": (
                "data/processed/classical/test_chain_diagnostics.csv"
            ),
            "figure_dir": "figures/exploratory/classical/day25/test",
        },
    }


def test_lean_schema_is_the_source_of_truth() -> None:
    parameters = validate_dataset_specification(tiny_specification())
    assert parameters["lattice_size"] == 4
    assert parameters["configurations_per_chain"] == 2
    assert parameters["seed_matrix"].shape == (2, 2)

    stale = tiny_specification()
    stale.pop("schema_version")
    stale["experiment"]["dataset_schema_version"] = "1.0"  # type: ignore[index]
    stale["simulation"] = stale.pop("dataset")
    with pytest.raises(ValueError, match="schema_version"):
        validate_dataset_specification(stale)


def test_configuration_chain_is_deterministic_and_stores_copies() -> None:
    first = sample_configuration_chain(
        lattice_size=4,
        temperature=2.0,
        burn_in_sweeps=2,
        production_sweeps=6,
        configuration_interval=2,
        seed=25,
        initial_state="up",
    )
    second = sample_configuration_chain(
        lattice_size=4,
        temperature=2.0,
        burn_in_sweeps=2,
        production_sweeps=6,
        configuration_interval=2,
        seed=25,
        initial_state="up",
    )

    configurations = first["configurations"]
    assert configurations.shape == (3, 4, 4)
    assert configurations.dtype == np.int8
    assert np.all(np.isin(configurations, (-1, 1)))
    assert np.array_equal(configurations, second["configurations"])
    assert not np.shares_memory(configurations[0], first["final_lattice"])

    saved = configurations.copy()
    first["final_lattice"][0, 0] *= -1
    assert np.array_equal(configurations, saved)


def test_dataset_integrity_and_spin_flip_symmetry() -> None:
    specification = tiny_specification()
    dataset = build_classical_ml_dataset(specification, verbose=False)
    report = validate_dataset(
        dataset,
        specification,
        previous_seeds=np.empty(0, dtype=np.int64),
    )

    assert dataset["configurations"].shape == (8, 4, 4)
    assert np.unique(dataset["chain_ids"]).size == 4
    assert np.unique(dataset["seeds"]).size == 4
    assert report["all_integrity_checks_passed"] is True

    configuration = dataset["configurations"][0]
    assert np.isclose(total_energy(configuration), total_energy(-configuration))
    assert np.isclose(magnetisation(-configuration), -magnetisation(configuration))


def test_save_dataset_writes_only_lean_outputs(tmp_path: Path) -> None:
    specification = tiny_specification()
    dataset = build_classical_ml_dataset(specification, verbose=False)
    saved = save_dataset(
        specification,
        dataset,
        project_root=tmp_path,
        overwrite=False,
    )

    expected_file_keys = {
        "dataset_npz",
        "manifest_csv",
        "summary_csv",
        "chain_diagnostics_csv",
    }
    assert expected_file_keys.issubset(saved["paths"])
    for key in expected_file_keys:
        assert saved["paths"][key].exists()

    assert not list(tmp_path.rglob("*metadata*.json"))
    assert not list(tmp_path.rglob("*CARD.md"))
    assert not list(tmp_path.rglob("*split_summary*.csv"))

    with np.load(saved["paths"]["dataset_npz"], allow_pickle=False) as reloaded:
        payload = serialisable_payload(dataset)
        assert set(reloaded.files) == set(payload)
        for name, values in payload.items():
            assert np.array_equal(reloaded[name], values)

    with pytest.raises(FileExistsError):
        save_dataset(
            specification,
            dataset,
            project_root=tmp_path,
            overwrite=False,
        )
