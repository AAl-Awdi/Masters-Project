"""Observable calculations for the two-dimensional classical Ising model.

The functions in this module contain no random-number generation and no file I/O.
Thus, they can be imported by scripts, notebooks and tests without hidden state.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


Array = np.ndarray


def _validate_spin_lattice(spins: Array) -> Array:
    """Return spins as an array after validating the lattice convention."""
    lattice = np.asarray(spins)

    if lattice.ndim != 2 or lattice.shape[0] != lattice.shape[1]:
        raise ValueError("spins must be a square two-dimensional array.")
    if lattice.shape[0] < 2:
        raise ValueError("the lattice side length must be at least two.")
    if not np.all(np.isin(lattice, (-1, 1))):
        raise ValueError("spins must contain only -1 and +1 values.")

    return lattice


def total_energy(
    spins: Array,
    coupling: float = 1.0,
    field: float = 0.0,
) -> float:
    r"""Return the total Ising energy with each bond counted exactly once.

    Periodic horizontal and vertical bonds are counted by pairing every site,
    with its right-hand and downward neighbours.
    This gives exactly 2 * L^2 bonds on an L x L square lattice.
    """
    lattice = _validate_spin_lattice(spins)

    right_products = lattice * np.roll(lattice, shift=-1, axis=1)
    down_products = lattice * np.roll(lattice, shift=-1, axis=0)

    interaction = -float(coupling) * (
        np.sum(right_products, dtype=np.int64)
        + np.sum(down_products, dtype=np.int64)
    )
    field_term = -float(field) * np.sum(lattice, dtype=np.int64)

    return float(interaction + field_term)


def total_energy_reference(
    spins: Array,
    coupling: float = 1.0,
    field: float = 0.0,
) -> float:
    """Transparent loop implementation used as an independent test oracle."""
    lattice = _validate_spin_lattice(spins)
    L = lattice.shape[0]
    energy = 0.0

    for row in range(L):
        for column in range(L):
            spin = int(lattice[row, column])
            energy -= coupling * spin * int(lattice[row, (column + 1) % L])
            energy -= coupling * spin * int(lattice[(row + 1) % L, column])
            energy -= field * spin

    return float(energy)


def energy_per_spin(
    spins: Array,
    coupling: float = 1.0,
    field: float = 0.0,
) -> float:
    """Return the total energy divided by the number of lattice sites."""
    lattice = _validate_spin_lattice(spins)
    return total_energy(lattice, coupling=coupling, field=field) / lattice.size


def magnetisation(spins: Array) -> float:
    """Return the signed magnetisation per spin."""
    lattice = _validate_spin_lattice(spins)
    return float(np.mean(lattice))


def absolute_magnetisation(spins: Array) -> float:
    """Return the absolute magnetisation per spin of one configuration."""
    return abs(magnetisation(spins))


def susceptibility(
    magnetisation_samples: Array,
    temperature: float,
    n_spins: int,
    boltzmann_constant: float = 1.0,
    centre: Literal["signed", "absolute"] = "signed",
) -> float:
    r"""Estimate the magnetic susceptibility per spin from sampled m values."""
    samples = np.asarray(magnetisation_samples, dtype=float)

    if samples.ndim != 1 or samples.size < 2:
        raise ValueError("magnetisation_samples must be a one-dimensional array.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if n_spins < 1:
        raise ValueError("n_spins must be positive.")
    if boltzmann_constant <= 0.0:
        raise ValueError("boltzmann_constant must be positive.")
    if centre not in {"signed", "absolute"}:
        raise ValueError("centre must be 'signed' or 'absolute'.")

    mean_square = float(np.mean(samples**2))
    if centre == "signed":
        centre_value = float(np.mean(samples))
    else:
        centre_value = float(np.mean(np.abs(samples)))

    variance_term = max(mean_square - centre_value**2, 0.0)
    return n_spins * variance_term / (boltzmann_constant * temperature)


def specific_heat(
    energy_samples: Array,
    temperature: float,
    n_spins: int,
    boltzmann_constant: float = 1.0,
) -> float:
    r"""Estimate the constant-volume specific heat per spin from total energies."""
    samples = np.asarray(energy_samples, dtype=float)

    if samples.ndim != 1 or samples.size < 2:
        raise ValueError("energy_samples must be a one-dimensional array.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if n_spins < 1:
        raise ValueError("n_spins must be positive.")
    if boltzmann_constant <= 0.0:
        raise ValueError("boltzmann_constant must be positive.")

    variance = max(
        float(np.mean(samples**2) - np.mean(samples) ** 2),
        0.0,
    )
    return variance / (n_spins * boltzmann_constant * temperature**2)


def exact_square_lattice_critical_temperature(
    coupling: float = 1.0,
    boltzmann_constant: float = 1.0,
) -> float:
    r"""Return the exact infinite-lattice zero-field square-Ising benchmark."""
    if coupling <= 0.0:
        raise ValueError("coupling must be positive for the ferromagnetic benchmark.")
    if boltzmann_constant <= 0.0:
        raise ValueError("boltzmann_constant must be positive.")

    return float(
        2.0 * coupling
        / (boltzmann_constant * np.log(1.0 + np.sqrt(2.0)))
    )
