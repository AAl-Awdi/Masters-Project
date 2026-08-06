"""Physics-facing tests for classical Ising observables."""

from __future__ import annotations

import numpy as np
import pytest

from ising2d.observables import (
    absolute_magnetisation,
    energy_per_spin,
    exact_square_lattice_critical_temperature,
    magnetisation,
    specific_heat,
    susceptibility,
    total_energy,
    total_energy_reference,
)


@pytest.mark.parametrize("lattice_size", [2, 4, 8])
def test_ordered_lattices_have_energy_per_spin_minus_two(
    lattice_size: int,
) -> None:
    all_up = np.ones((lattice_size, lattice_size), dtype=np.int8)
    all_down = -all_up

    assert energy_per_spin(all_up) == pytest.approx(-2.0)
    assert energy_per_spin(all_down) == pytest.approx(-2.0)


@pytest.mark.parametrize("lattice_size", [2, 4, 8])
def test_even_checkerboard_has_energy_per_spin_plus_two(
    lattice_size: int,
) -> None:
    row, column = np.indices((lattice_size, lattice_size))
    checkerboard = np.where((row + column) % 2 == 0, 1, -1).astype(
        np.int8
    )

    assert energy_per_spin(checkerboard) == pytest.approx(2.0)


def test_vectorised_energy_matches_reference_loop() -> None:
    rng = np.random.default_rng(1901)

    for lattice_size in (2, 3, 7):
        for _ in range(5):
            lattice = rng.choice(
                (-1, 1),
                size=(lattice_size, lattice_size),
            ).astype(np.int8)
            assert total_energy(lattice) == pytest.approx(
                total_energy_reference(lattice)
            )


def test_magnetisation_definitions() -> None:
    lattice = np.array([[1, 1], [1, -1]], dtype=np.int8)

    assert magnetisation(lattice) == pytest.approx(0.5)
    assert absolute_magnetisation(lattice) == pytest.approx(0.5)
    assert magnetisation(-lattice) == pytest.approx(-0.5)
    assert absolute_magnetisation(-lattice) == pytest.approx(0.5)


def test_fluctuation_observables_vanish_for_constant_samples() -> None:
    energy_samples = np.full(20, -32.0)
    magnetisation_samples = np.full(20, 0.75)

    assert specific_heat(
        energy_samples,
        temperature=2.0,
        n_spins=16,
    ) == pytest.approx(0.0)
    assert susceptibility(
        magnetisation_samples,
        temperature=2.0,
        n_spins=16,
    ) == pytest.approx(0.0)


def test_signed_and_absolute_susceptibility_conventions_are_distinct() -> None:
    samples = np.array([-1.0, 1.0, -1.0, 1.0])

    signed = susceptibility(
        samples,
        temperature=2.0,
        n_spins=4,
        centre="signed",
    )
    absolute_centred = susceptibility(
        samples,
        temperature=2.0,
        n_spins=4,
        centre="absolute",
    )

    assert signed == pytest.approx(2.0)
    assert absolute_centred == pytest.approx(0.0)


def test_exact_square_lattice_benchmark() -> None:
    assert exact_square_lattice_critical_temperature() == pytest.approx(
        2.269185314213022,
        rel=1e-14,
    )
