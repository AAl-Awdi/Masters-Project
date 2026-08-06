"""Reproducible local-Metropolis simulation for the 2D classical Ising model."""

from __future__ import annotations

import numpy as np

from .observables import magnetisation, total_energy


Array = np.ndarray


def initialise_lattice(
    lattice_size: int,
    rng: np.random.Generator,
    mode: str = "random",
) -> Array:
    """Create a square lattice containing only -1 and +1."""
    if lattice_size < 2:
        raise ValueError("lattice_size must be at least two.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an instance of numpy.random.Generator.")

    shape = (lattice_size, lattice_size)

    if mode == "random":
        return rng.choice((-1, 1), size=shape).astype(np.int8)
    if mode == "up":
        return np.ones(shape, dtype=np.int8)
    if mode == "down":
        return -np.ones(shape, dtype=np.int8)

    raise ValueError("mode must be 'random', 'up' or 'down'.")


def local_energy_change(
    spins: Array,
    row: int,
    column: int,
    coupling: float = 1.0,
    field: float = 0.0,
) -> float:
    r"""Return the energy change for flipping one spin."""
    lattice = np.asarray(spins)
    if lattice.ndim != 2 or lattice.shape[0] != lattice.shape[1]:
        raise ValueError("spins must be a square two-dimensional array.")

    L = lattice.shape[0]
    if not (0 <= row < L and 0 <= column < L):
        raise IndexError("row and column must refer to a lattice site.")

    spin = int(lattice[row, column])
    neighbour_sum = (
        int(lattice[(row - 1) % L, column])
        + int(lattice[(row + 1) % L, column])
        + int(lattice[row, (column - 1) % L])
        + int(lattice[row, (column + 1) % L])
    )

    return float(2.0 * spin * (coupling * neighbour_sum + field))


def metropolis_attempt(
    spins: Array,
    row: int,
    column: int,
    temperature: float,
    rng: np.random.Generator,
    coupling: float = 1.0,
    field: float = 0.0,
    boltzmann_constant: float = 1.0,
) -> bool:
    """Attempt one local spin flip and report whether it was accepted."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if boltzmann_constant <= 0.0:
        raise ValueError("boltzmann_constant must be positive.")

    delta_energy = local_energy_change(
        spins,
        row,
        column,
        coupling=coupling,
        field=field,
    )

    if delta_energy <= 0.0:
        spins[row, column] *= -1
        return True

    probability = np.exp(
        -delta_energy / (boltzmann_constant * temperature)
    )
    if rng.random() < probability:
        spins[row, column] *= -1
        return True

    return False


def metropolis_sweep(
    spins: Array,
    temperature: float,
    rng: np.random.Generator,
    coupling: float = 1.0,
    field: float = 0.0,
    boltzmann_constant: float = 1.0,
) -> float:
    """Perform exactly L^2 random attempts with replacement.

    The validated public single-attempt function is intentionally not called in this inner loop.
    Inlining the small local calculation avoids millions of repeated array validations and Python function calls,
    while preserving the same random-number order and Metropolis rule.
    """
    lattice = np.asarray(spins)
    if lattice.ndim != 2 or lattice.shape[0] != lattice.shape[1]:
        raise ValueError("spins must be a square two-dimensional array.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if boltzmann_constant <= 0.0:
        raise ValueError("boltzmann_constant must be positive.")

    L = lattice.shape[0]
    accepted = 0
    inverse_thermal_energy = 1.0 / (boltzmann_constant * temperature)

    # Local bindings reduce attribute-lookup overhead inside the hot loop.
    integers = rng.integers
    random = rng.random
    J = float(coupling)
    B = float(field)

    for _ in range(lattice.size):
        row = int(integers(0, L))
        column = int(integers(0, L))
        spin = int(lattice[row, column])
        neighbour_sum = (
            int(lattice[(row - 1) % L, column])
            + int(lattice[(row + 1) % L, column])
            + int(lattice[row, (column - 1) % L])
            + int(lattice[row, (column + 1) % L])
        )
        delta_energy = 2.0 * spin * (J * neighbour_sum + B)

        if delta_energy <= 0.0 or random() < np.exp(
            -delta_energy * inverse_thermal_energy
        ):
            lattice[row, column] *= -1
            accepted += 1

    return accepted / lattice.size


def run_chain(
    lattice_size: int,
    temperature: float,
    burn_in_sweeps: int,
    production_sweeps: int,
    sample_every: int,
    seed: int,
    initial_state: str = "up",
    coupling: float = 1.0,
    field: float = 0.0,
    boltzmann_constant: float = 1.0,
    record_burn_in: bool = True,
) -> dict[str, object]:
    """Run one independently seeded fixed-temperature Markov chain.

    production_sweeps is the number of full Metropolis sweeps performed after burn-in.
    A sample is retained after every sample_every sweeps.
    To keep output shapes unambiguous, production_sweeps must be divisible by sample_every.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if burn_in_sweeps < 0:
        raise ValueError("burn_in_sweeps must be non-negative.")
    if production_sweeps < 1:
        raise ValueError("production_sweeps must be positive.")
    if sample_every < 1:
        raise ValueError("sample_every must be at least one.")
    if production_sweeps % sample_every != 0:
        raise ValueError(
            "production_sweeps must be divisible by sample_every."
        )

    rng = np.random.default_rng(seed)
    lattice = initialise_lattice(lattice_size, rng, mode=initial_state)
    initial_lattice = lattice.copy()

    if record_burn_in:
        burn_energy = np.empty(burn_in_sweeps + 1, dtype=float)
        burn_magnetisation = np.empty(burn_in_sweeps + 1, dtype=float)
        burn_acceptance = np.empty(burn_in_sweeps, dtype=float)
        burn_energy[0] = total_energy(
            lattice,
            coupling=coupling,
            field=field,
        )
        burn_magnetisation[0] = magnetisation(lattice)
    else:
        burn_energy = np.empty(0, dtype=float)
        burn_magnetisation = np.empty(0, dtype=float)
        burn_acceptance = np.empty(0, dtype=float)

    for sweep_index in range(burn_in_sweeps):
        acceptance = metropolis_sweep(
            lattice,
            temperature,
            rng,
            coupling=coupling,
            field=field,
            boltzmann_constant=boltzmann_constant,
        )
        if record_burn_in:
            burn_acceptance[sweep_index] = acceptance
            burn_energy[sweep_index + 1] = total_energy(
                lattice,
                coupling=coupling,
                field=field,
            )
            burn_magnetisation[sweep_index + 1] = magnetisation(lattice)

    n_samples = production_sweeps // sample_every
    production_energy = np.empty(n_samples, dtype=float)
    production_magnetisation = np.empty(n_samples, dtype=float)
    production_acceptance = np.empty(n_samples, dtype=float)
    production_sweep_numbers = np.empty(n_samples, dtype=np.int64)

    acceptance_accumulator = 0.0
    sample_index = 0

    for sweep_index in range(1, production_sweeps + 1):
        acceptance_accumulator += metropolis_sweep(
            lattice,
            temperature,
            rng,
            coupling=coupling,
            field=field,
            boltzmann_constant=boltzmann_constant,
        )

        if sweep_index % sample_every == 0:
            production_energy[sample_index] = total_energy(
                lattice,
                coupling=coupling,
                field=field,
            )
            production_magnetisation[sample_index] = magnetisation(lattice)
            production_acceptance[sample_index] = (
                acceptance_accumulator / sample_every
            )
            production_sweep_numbers[sample_index] = sweep_index
            acceptance_accumulator = 0.0
            sample_index += 1

    production_energy_squared = production_energy**2
    production_magnetisation_squared = production_magnetisation**2
    production_absolute_magnetisation = np.abs(production_magnetisation)
    n_spins = lattice_size**2

    return {
        "lattice_size": int(lattice_size),
        "n_spins": int(n_spins),
        "temperature": float(temperature),
        "coupling": float(coupling),
        "field": float(field),
        "boltzmann_constant": float(boltzmann_constant),
        "seed": int(seed),
        "initial_state": str(initial_state),
        "burn_in_sweeps": int(burn_in_sweeps),
        "production_sweeps": int(production_sweeps),
        "sample_every": int(sample_every),
        "record_burn_in": bool(record_burn_in),
        "initial_lattice": initial_lattice,
        "final_lattice": lattice.copy(),
        "burn_energy": burn_energy,
        "burn_magnetisation": burn_magnetisation,
        "burn_acceptance": burn_acceptance,
        "production_sweep_numbers": production_sweep_numbers,
        "production_energy": production_energy,
        "production_energy_squared": production_energy_squared,
        "production_magnetisation": production_magnetisation,
        "production_magnetisation_squared": (
            production_magnetisation_squared
        ),
        "production_absolute_magnetisation": (
            production_absolute_magnetisation
        ),
        "production_acceptance": production_acceptance,
        "mean_total_energy": float(np.mean(production_energy)),
        "mean_energy_squared": float(
            np.mean(production_energy_squared)
        ),
        "mean_energy_per_spin": float(
            np.mean(production_energy) / n_spins
        ),
        "mean_signed_magnetisation": float(
            np.mean(production_magnetisation)
        ),
        "mean_magnetisation_squared": float(
            np.mean(production_magnetisation_squared)
        ),
        "mean_absolute_magnetisation": float(
            np.mean(production_absolute_magnetisation)
        ),
        "mean_acceptance_fraction": float(
            np.mean(production_acceptance)
        ),
    }
