"""Dataset, seed and serialisation tests for multi-chain experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ising2d.dataset import (
    load_npz,
    make_seed_matrix,
    run_temperature_sweep,
    save_temperature_sweep,
    summary_rows,
    validate_temperature_grid,
)


SWEEP_ARRAY_FIELDS = (
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


def _small_sweep() -> dict[str, object]:
    temperatures = np.array([1.5, 2.269185314213022, 3.5])
    return run_temperature_sweep(
        temperatures=temperatures,
        seeds=make_seed_matrix(26000, 2, temperatures.size),
        lattice_size=4,
        coupling=1.0,
        field=0.0,
        boltzmann_constant=1.0,
        burn_in_sweeps=6,
        production_sweeps=12,
        sample_every=3,
        initial_state="up",
        record_burn_in=True,
        verbose=False,
    )


def test_temperature_grid_validation_is_used_by_the_pipeline() -> None:
    assert np.array_equal(
        validate_temperature_grid([1.5, 2.0, 3.5]),
        np.array([1.5, 2.0, 3.5]),
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        validate_temperature_grid([1.5, 2.0, 2.0])


def test_day23_seed_ranges_are_unique_and_disjoint_from_day22() -> None:
    day22 = (
        set(range(21900, 21903))
        | set(range(22000, 22018))
        | {22901, 22902, 22903}
    )
    day23_main = make_seed_matrix(23000, 5, 18)
    day23_diagnostics = {23100, 23101, 23102}
    day23_smoke = set(make_seed_matrix(23200, 2, 3).ravel().tolist())

    all_day23 = (
        set(day23_main.ravel().tolist())
        | day23_diagnostics
        | day23_smoke
    )

    assert day23_main.shape == (5, 18)
    assert np.unique(day23_main).size == 90
    assert not (day22 & all_day23)
    assert len(all_day23) == 90 + 3 + 6


def test_multi_replicate_sweep_shapes_moments_and_reproducibility() -> None:
    first = _small_sweep()
    second = _small_sweep()

    assert np.asarray(first["seeds"]).shape == (2, 3)
    assert np.asarray(first["production_energy"]).shape == (2, 3, 4)
    assert np.asarray(first["production_magnetisation"]).shape == (2, 3, 4)
    assert np.asarray(first["burn_energy"]).shape == (2, 3, 7)
    assert np.asarray(first["initial_lattice"]).shape == (2, 3, 4, 4)

    for field in SWEEP_ARRAY_FIELDS:
        assert np.array_equal(first[field], second[field])

    energy = np.asarray(first["production_energy"], dtype=float)
    magnetisation = np.asarray(
        first["production_magnetisation"],
        dtype=float,
    )
    n_spins = int(first["n_spins"])

    np.testing.assert_allclose(
        first["mean_total_energy"],
        np.mean(energy, axis=2),
    )
    np.testing.assert_allclose(
        first["mean_energy_squared"],
        np.mean(energy**2, axis=2),
    )
    np.testing.assert_allclose(
        first["mean_energy_per_spin"],
        np.mean(energy, axis=2) / n_spins,
    )
    np.testing.assert_allclose(
        first["mean_signed_magnetisation"],
        np.mean(magnetisation, axis=2),
    )
    np.testing.assert_allclose(
        first["mean_magnetisation_squared"],
        np.mean(magnetisation**2, axis=2),
    )
    np.testing.assert_allclose(
        first["mean_absolute_magnetisation"],
        np.mean(np.abs(magnetisation), axis=2),
    )


def test_summary_contains_one_auditable_row_per_chain() -> None:
    sweep = _small_sweep()
    rows = summary_rows(sweep)

    assert len(rows) == 2 * 3
    assert {(row["replicate"], row["temperature"]) for row in rows} == {
        (replicate, temperature)
        for replicate in (0, 1)
        for temperature in (1.5, 2.269185314213022, 3.5)
    }


def test_save_reload_preserves_arrays_and_records_multiseed_metadata(
    tmp_path: Path,
) -> None:
    sweep = _small_sweep()
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("experiment: test\n", encoding="utf-8")
    config = {
        "experiment": {"name": "test_multiseed", "description": "test"},
        "model": {"lattice_size": 4},
        "simulation": {"n_replicates": 2},
        "outputs": {},
    }

    result_path = tmp_path / "result.npz"
    summary_path = tmp_path / "summary.csv"
    metadata_path = tmp_path / "metadata.json"
    save_temperature_sweep(
        sweep,
        result_path=result_path,
        summary_path=summary_path,
        metadata_path=metadata_path,
        config=config,
        config_path=config_path,
    )

    loaded = load_npz(result_path)
    for field, saved_array in loaded.items():
        if field in sweep and field != "runs":
            assert np.array_equal(saved_array, np.asarray(sweep[field]))

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    assert metadata["seed_matrix"] == np.asarray(sweep["seeds"]).tolist()
    assert metadata["array_shapes"]["production_energy"] == [2, 3, 4]
    assert "day-22" not in " ".join(metadata["limitations"]).lower()
    assert "2 independently seeded chain(s)" in metadata["limitations"][0]
