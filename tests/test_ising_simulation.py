"""Reproducibility and consistency tests for the Metropolis simulation."""

from __future__ import annotations

import numpy as np
import pytest

from ising2d.dataset import make_seed_matrix, run_temperature_sweep
from ising2d.observables import total_energy
from ising2d.simulation import (
    initialise_lattice,
    local_energy_change,
    metropolis_sweep,
    run_chain,
)


def test_initialisation_is_reproducible() -> None:
    first = initialise_lattice(8, np.random.default_rng(123), "random")
    second = initialise_lattice(8, np.random.default_rng(123), "random")
    assert np.array_equal(first, second)


def test_local_delta_energy_matches_global_energy_difference() -> None:
    rng = np.random.default_rng(1902)
    lattice = initialise_lattice(8, rng, "random")

    for row, column in ((0, 0), (2, 5), (7, 7)):
        before = total_energy(lattice)
        predicted = local_energy_change(lattice, row, column)
        flipped = lattice.copy()
        flipped[row, column] *= -1
        after = total_energy(flipped)

        assert after - before == pytest.approx(predicted)


def test_one_sweep_returns_a_physical_acceptance_fraction() -> None:
    lattice = np.ones((8, 8), dtype=np.int8)
    acceptance = metropolis_sweep(
        lattice,
        temperature=2.5,
        rng=np.random.default_rng(1234),
    )

    assert 0.0 <= acceptance <= 1.0
    assert np.all(np.isin(lattice, (-1, 1)))


def test_run_chain_reproduces_all_arrays() -> None:
    arguments = {
        "lattice_size": 8,
        "temperature": 2.5,
        "burn_in_sweeps": 20,
        "production_sweeps": 40,
        "sample_every": 4,
        "seed": 12345,
        "initial_state": "random",
    }
    first = run_chain(**arguments)
    second = run_chain(**arguments)

    for field in (
        "initial_lattice",
        "final_lattice",
        "burn_energy",
        "burn_magnetisation",
        "production_energy",
        "production_energy_squared",
        "production_magnetisation",
        "production_magnetisation_squared",
        "production_absolute_magnetisation",
        "production_acceptance",
    ):
        assert np.array_equal(first[field], second[field])

    assert np.asarray(first["burn_energy"]).shape == (21,)
    assert np.asarray(first["production_energy"]).shape == (10,)
    assert np.array_equal(
        first["production_sweep_numbers"],
        np.array([4, 8, 12, 16, 20, 24, 28, 32, 36, 40]),
    )


def test_seed_matrix_is_unique_and_aligned() -> None:
    seeds = make_seed_matrix(22000, n_replicates=2, n_temperatures=3)
    assert seeds.shape == (2, 3)
    assert np.array_equal(
        seeds,
        np.array([[22000, 22001, 22002], [22003, 22004, 22005]]),
    )
    assert np.unique(seeds).size == seeds.size


def test_temperature_sweep_shapes_and_physical_trends() -> None:
    temperatures = np.array([1.5, 3.5])
    seeds = make_seed_matrix(25000, 1, temperatures.size)

    sweep = run_temperature_sweep(
        temperatures=temperatures,
        seeds=seeds,
        lattice_size=8,
        coupling=1.0,
        field=0.0,
        boltzmann_constant=1.0,
        burn_in_sweeps=100,
        production_sweeps=200,
        sample_every=1,
        initial_state="up",
        record_burn_in=True,
        verbose=False,
    )

    assert np.asarray(sweep["production_energy"]).shape == (1, 2, 200)
    assert np.asarray(sweep["production_magnetisation"]).shape == (
        1,
        2,
        200,
    )
    assert np.asarray(sweep["mean_energy_per_spin"]).shape == (1, 2)
    assert np.asarray(sweep["mean_absolute_magnetisation"]).shape == (
        1,
        2,
    )

    mean_energy = np.asarray(sweep["mean_energy_per_spin"])[0]
    mean_absolute_magnetisation = np.asarray(
        sweep["mean_absolute_magnetisation"]
    )[0]

    assert mean_energy[0] < mean_energy[1]
    assert mean_absolute_magnetisation[0] > mean_absolute_magnetisation[1]
    assert np.all((-2.0 <= mean_energy) & (mean_energy <= 2.0))
