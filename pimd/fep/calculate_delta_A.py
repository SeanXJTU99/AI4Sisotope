"""Calculate Helmholtz free energy differences for isotope substitution
in both liquid and gas phases from PIMD-FEP trajectories.

Orchestrates the per-phase FEP calculation and computes the phase
differential DeltaDeltaA = DeltaA_liquid - DeltaA_gas, which
determines the direction and magnitude of isotope fractionation.

A positive DeltaDeltaA means the heavy isotope is more stable in
the gas phase -> inverse isotope effect (enriched at column top).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from pimd.fep.fep_mass_mutation import (
    IsotopePair,
    compute_fep_free_energy,
    compute_spring_energy_difference,
    run_mass_mutation_fep,
    KB,
)


@dataclass
class PhaseConfig:
    """Configuration for a single-phase PIMD-FEP calculation.

    Attributes:
        phase: 'liquid' or 'gas'.
        pimd_traj: Path to i-PI trajectory output (.xyz format).
        temperature: Simulation temperature in K.
        n_beads: Number of PIMD beads used.
        density: Density in g/cm3 (for reference, liquid only).
    """
    phase: str
    pimd_traj: Path
    temperature: float
    n_beads: int
    density: Optional[float] = None


def load_pimd_trajectory(
    traj_path: Path,
    n_atoms: int = 4,
    n_beads: int = 64,
) -> np.ndarray:
    """Load an i-PI XYZ trajectory into a numpy array.

    i-PI writes multi-structure XYZ files where the comment line
    contains bead index information. Each frame spans (n_atoms * n_beads)
    coordinate lines.

    Args:
        traj_path: Path to i-PI trajectory file.
        n_atoms: Number of atoms per replica.
        n_beads: Number of PIMD beads.

    Returns:
        Bead positions, shape (n_frames, n_atoms, n_beads, 3) in Angstrom.

    Raises:
        FileNotFoundError: If trajectory file does not exist.
    """
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory not found: {traj_path}")

    # Mock trajectory loading -- in production parses i-PI XYZ format
    rng = np.random.default_rng(hash(str(traj_path)) % (2**31))
    n_frames = 1000

    # Generate realistic mock trajectory around BF3 equilibrium
    coords = rng.normal(0, 0.05, size=(n_frames, n_atoms, n_beads, 3))
    # B at center
    coords[:, 0, :, :] += np.array([0.0, 0.0, 0.0])[None, None, :]
    # Three F atoms at ~1.3 A
    coords[:, 1, :, :] += np.array([1.3, 0.0, 0.0])[None, None, :]
    coords[:, 2, :, :] += np.array([-0.65, 1.13, 0.0])[None, None, :]
    coords[:, 3, :, :] += np.array([-0.65, -1.13, 0.0])[None, None, :]

    return coords


def compute_phase_delta_a(
    config: PhaseConfig,
    isotope_pair: IsotopePair,
) -> dict[str, float]:
    """Compute FEP free energy for a single phase.

    Args:
        config: Phase configuration (liquid or gas).
        isotope_pair: Isotope substitution specification.

    Returns:
        Dict with delta_A, delta_A_std, and phase metadata.
    """
    traj = load_pimd_trajectory(
        config.pimd_traj,
        n_beads=config.n_beads,
    )

    result = run_mass_mutation_fep(
        traj,
        isotope_pair,
        temperature=config.temperature,
        n_beads=config.n_beads,
    )

    result["phase"] = config.phase
    result["density_g_cm3"] = config.density
    return result


def compute_delta_delta_a(
    liquid_result: dict[str, float],
    gas_result: dict[str, float],
) -> dict[str, float]:
    """Compute liquid-gas free energy differential.

    DeltaDeltaA = DeltaA_liquid - DeltaA_gas

    The sign determines fractionation direction:
        DeltaDeltaA > 0 -> heavy isotope prefers gas phase (inverse effect)
        DeltaDeltaA < 0 -> heavy isotope prefers liquid phase (normal effect)
        DeltaDeltaA ~ 0 -> no fractionation

    Args:
        liquid_result: FEP result dict for liquid phase.
        gas_result: FEP result dict for gas phase.

    Returns:
        Dict with delta_delta_A and propagated uncertainties.
    """
    dda = liquid_result["delta_A_eV"] - gas_result["delta_A_eV"]
    dda_std = np.sqrt(
        liquid_result["delta_A_std_eV"] ** 2 + gas_result["delta_A_std_eV"] ** 2
    )

    return {
        "delta_delta_A_eV": round(dda, 8),
        "delta_delta_A_std_eV": round(dda_std, 8),
        "delta_delta_A_meV": round(dda * 1000, 5),
        "delta_delta_A_std_meV": round(dda_std * 1000, 5),
        "temperature_K": liquid_result["temperature_K"],
        "n_beads": liquid_result["n_beads"],
    }


def interpret_fractionation(
    delta_delta_a: float,
    delta_delta_a_std: float,
    temperature: float,
) -> str:
    """Interpret the sign and magnitude of DeltaDeltaA.

    Args:
        delta_delta_a: Free energy differential in eV.
        delta_delta_a_std: Uncertainty in eV.
        temperature: Temperature in K.

    Returns:
        Human-readable interpretation string.
    """
    kt = KB * temperature
    confidence = abs(delta_delta_a) / max(delta_delta_a_std, 1e-12)

    if confidence < 2.0:
        return (
            f"Uncertain (DeltaDeltaA = {delta_delta_a*1000:.3f} +/- "
            f"{delta_delta_a_std*1000:.3f} meV, {confidence:.1f} sigma)"
        )

    if delta_delta_a > 0:
        return (
            f"Inverse isotope effect: heavy isotope enriched in gas phase "
            f"(top of distillation column). DeltaDeltaA = {delta_delta_a*1000:.3f} meV. "
            f"Confidence: {confidence:.1f} sigma."
        )
    else:
        return (
            f"Normal isotope effect: light isotope enriched in gas phase "
            f"(bottom of distillation column). DeltaDeltaA = {delta_delta_a*1000:.3f} meV. "
            f"Confidence: {confidence:.1f} sigma."
        )
