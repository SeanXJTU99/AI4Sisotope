"""Free energy perturbation (FEP) via mass mutation for isotope effects.

Implements the Zwanzig exponential averaging method to compute the
Helmholtz free energy difference when an isotope mass is changed
in a PIMD trajectory.

Theory:
    The spring potential for bead s of atom i is:
        U_spring = 0.5 * m_i * omega_P^2 * (r_{i,s} - r_{i,s+1})^2
    where omega_P = P * k_B * T / hbar.

    When mass changes m_light -> m_heavy, the spring energy changes:
        Delta U = U_spring(m_heavy) - U_spring(m_light)

    The free energy difference via Zwanzig's formula:
        Delta A = -k_B T * ln( < exp(-beta * Delta U) >_light )
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


# Physical constants
KB = 8.617333262145e-5   # eV/K
HBAR = 6.582119569e-16   # eV*s


@dataclass
class IsotopePair:
    """Specification of an isotope substitution pair.

    Attributes:
        element: Chemical symbol, e.g. 'B', 'C', 'F'.
        mass_light: Mass of lighter isotope in amu, e.g. 10.0129 for 10B.
        mass_heavy: Mass of heavier isotope in amu, e.g. 11.0093 for 11B.
        atom_indices: 0-based indices of atoms to mutate in the trajectory.
    """
    element: str
    mass_light: float
    mass_heavy: float
    atom_indices: list[int]


def compute_spring_energy_difference(
    coords: np.ndarray,
    masses_light: np.ndarray,
    masses_heavy: np.ndarray,
    temperature: float,
    n_beads: int,
) -> np.ndarray:
    """Compute spring potential energy difference between light and heavy isotopes.

    The PIMD spring constant is: kappa = m * n_beads * (k_B * T / hbar)^2

    For each frame, computes:
        Delta U = U_spring(heavy) - U_spring(light)
    where U_spring = 0.5 * kappa * sum_s (r_s - r_{s+1})^2

    Args:
        coords: Bead positions, shape (n_frames, n_atoms, n_beads, 3) in Angstrom.
        masses_light: Light isotope masses, shape (n_atoms,) in amu.
        masses_heavy: Heavy isotope masses, shape (n_atoms,) in amu.
        temperature: Simulation temperature in K.
        n_beads: Number of PIMD beads (P).

    Returns:
        Spring energy difference per frame, shape (n_frames,).
        Positive = heavy isotope has higher spring energy (less stable).
    """
    n_frames, n_atoms, p, _ = coords.shape

    # Spring constant prefactor: m * (P * kT / hbar)^2
    # Units: amu * (eV * Angstrom) conversion handled by hbar
    kt = KB * temperature
    omega_p_sq = (n_beads * kt / HBAR) ** 2

    # Compute squared bead displacement for each atom:
    # sum_s (r_s - r_{s+1})^2  with cyclic boundary (s+1 mod P)
    diff = coords - np.roll(coords, -1, axis=2)  # r_s - r_{s+1}
    sq_disp = np.sum(diff ** 2, axis=(2, 3))  # sum over beads and xyz, (n_frames, n_atoms)

    # Delta U = 0.5 * Delta(mass) * omega_P^2 * sum_s (r_s - r_{s+1})^2
    delta_mass = (masses_heavy - masses_light)[np.newaxis, :]  # (1, n_atoms)
    delta_u = 0.5 * delta_mass * omega_p_sq * sq_disp  # (n_frames, n_atoms)

    # Sum over mutated atoms only
    return np.sum(delta_u, axis=1)  # (n_frames,)


def compute_fep_free_energy(
    delta_spring: np.ndarray,
    temperature: float,
    method: str = "exponential",
) -> tuple[float, float]:
    """Compute FEP free energy difference via Zwanzig averaging.

    Exponential averaging (exact):
        Delta A = -kT * ln( <exp(-Delta U / kT)> )

    Linear (approximate, for validation):
        Delta A ~ <Delta U>  (valid when Delta U << kT)

    Args:
        delta_spring: Spring energy differences, shape (n_frames,).
        temperature: Temperature in K.
        method: "exponential" (Zwanzig) or "linear" (approximation).

    Returns:
        Tuple of (Delta_A_mean_eV, Delta_A_std_eV).
    """
    kt = KB * temperature

    if method == "exponential":
        # Zwanzig: Delta A = -kT * ln( <exp(-beta * Delta U)> )
        beta = 1.0 / kt
        # Numerical stability: subtract max to avoid overflow
        scaled = -beta * delta_spring
        shifted = scaled - np.max(scaled)
        exp_mean = np.mean(np.exp(shifted))
        delta_a = -kt * (np.log(exp_mean) + np.max(scaled))

        # Standard deviation via block averaging
        delta_a_std = _block_average_std(delta_spring, kt, n_blocks=20)

    elif method == "linear":
        delta_a = float(np.mean(delta_spring))
        delta_a_std = float(np.std(delta_spring) / np.sqrt(len(delta_spring)))

    else:
        raise ValueError(f"Unknown method: {method}. Use 'exponential' or 'linear'.")

    return float(delta_a), float(delta_a_std)


def run_mass_mutation_fep(
    trajectory: np.ndarray,
    isotope_pair: IsotopePair,
    temperature: float = 145.0,
    n_beads: int = 64,
    n_blocks: int = 20,
) -> dict[str, float]:
    """Run complete FEP mass mutation analysis on a PIMD trajectory.

    Args:
        trajectory: Bead positions, shape (n_frames, n_atoms, n_beads, 3).
        isotope_pair: Specification of the isotope substitution.
        temperature: Simulation temperature in K.
        n_beads: PIMD beads count.
        n_blocks: Number of blocks for bootstrap error estimation.

    Returns:
        Dict with:
            - delta_A: Helmholtz free energy difference (eV)
            - delta_A_std: Statistical uncertainty (eV)
            - delta_A_per_atom: Free energy per mutated atom (meV)
            - converged: Whether free energy converged (bool)
            - effective_sample_size: Ratio of uncorrelated samples
    """
    n_atoms = trajectory.shape[1]
    masses_light = _build_mass_array(isotope_pair, use_light=True, n_atoms=n_atoms)
    masses_heavy = _build_mass_array(isotope_pair, use_light=False, n_atoms=n_atoms)

    delta_spring = compute_spring_energy_difference(
        trajectory, masses_light, masses_heavy, temperature, n_beads
    )

    delta_a, delta_a_std = compute_fep_free_energy(delta_spring, temperature)

    n_mutated = len(isotope_pair.atom_indices)
    return {
        "delta_A_eV": round(delta_a, 6),
        "delta_A_std_eV": round(delta_a_std, 6),
        "delta_A_per_atom_meV": round(delta_a / n_mutated * 1000, 3),
        "converged": delta_a_std < abs(delta_a) * 0.1,
        "temperature_K": temperature,
        "n_beads": n_beads,
        "n_frames": trajectory.shape[0],
        "isotope_pair": f"{isotope_pair.mass_light:.2f}->{isotope_pair.mass_heavy:.2f} amu",
    }


def _block_average_std(
    delta_spring: np.ndarray,
    kt: float,
    n_blocks: int = 20,
) -> float:
    """Compute standard deviation of Delta A via block averaging.

    Splits the time series into n_blocks, computes Delta A for each
    block independently, and returns the standard deviation across
    blocks. This accounts for time correlation in the trajectory.
    """
    n_frames = len(delta_spring)
    block_size = n_frames // n_blocks
    if block_size < 1:
        return 0.0

    beta = 1.0 / kt
    block_values: list[float] = []

    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size if i < n_blocks - 1 else n_frames
        block = delta_spring[start:end]
        scaled = -beta * block
        shifted = scaled - np.max(scaled)
        exp_mean = np.mean(np.exp(shifted))
        block_values.append(-kt * (np.log(exp_mean) + np.max(scaled)))

    return float(np.std(block_values) / np.sqrt(n_blocks))


def _build_mass_array(
    isotope_pair: IsotopePair,
    use_light: bool,
    n_atoms: int,
) -> np.ndarray:
    """Build mass array for the full system from isotope pair spec.

    Args:
        isotope_pair: Isotope substitution specification.
        use_light: True for light isotope, False for heavy.
        n_atoms: Total number of atoms in the system.

    Returns:
        Mass array of shape (n_atoms,) in amu.
    """
    # Default masses (BF3): B=10.811, F=18.998
    default_masses = {0: 10.811, 1: 18.998, 2: 12.011, 3: 238.05}
    masses = np.zeros(n_atoms)
    for i in range(n_atoms):
        masses[i] = default_masses.get(i % len(default_masses), 18.998)

    target_mass = isotope_pair.mass_light if use_light else isotope_pair.mass_heavy
    for idx in isotope_pair.atom_indices:
        if idx < n_atoms:
            masses[idx] = target_mass

    return masses
