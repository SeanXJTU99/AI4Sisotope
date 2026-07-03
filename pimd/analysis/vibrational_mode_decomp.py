"""Vibrational mode decomposition for PIMD trajectories.

Projects PIMD centroid positions onto normal mode displacement vectors
to quantify per-mode contributions to the isotope effect free energy.

For BF3 (D3h), the key modes are:
    nu1 (A1'): symmetric stretch  ~888 cm-1
    nu2 (A2''): out-of-plane bend ~691 cm-1  -- DOMINANT for IVPIE
    nu3 (E'):   asymmetric stretch ~1503 cm-1
    nu4 (E'):   in-plane bend       ~480 cm-1

The nu2 out-of-plane umbrella bending mode experiences severe spatial
hindrance in the condensed phase, causing a blue-shift that drives
the inverse vapor pressure isotope effect (IVPIE): the heavier 11B
isotope has lower ZPE penalty, making it more volatile.
"""

from pathlib import Path
from typing import Optional

import numpy as np


# Mode frequencies for target molecules (cm-1, from NIST/experiment)
MODE_DATA: dict[str, dict[str, tuple[float, str, np.ndarray]]] = {
    "BF3": {
        "nu1": (888.0, "A1' symmetric stretch",
                np.array([[0, 0, 0], [1, 0, 0], [-0.5, 0.866, 0], [-0.5, -0.866, 0]])),
        "nu2": (691.0, "A2'' out-of-plane bend (KEY: drives IVPIE)",
                np.array([[0, 0, 1], [0, 0, -0.333], [0, 0, -0.333], [0, 0, -0.333]])),
        "nu3": (1503.0, "E' asymmetric stretch (doubly degenerate)",
                np.array([[0, 0, 0], [1, 0, 0], [-0.5, 0, 0], [-0.5, 0, 0]])),
        "nu4": (480.0, "E' in-plane bend (doubly degenerate)",
                np.array([[0, 0, 0], [0, 1, 0], [0.866, -0.5, 0], [-0.866, -0.5, 0]])),
    },
    "CF4": {
        "nu1": (908.0, "A1 symmetric stretch",
                np.array([[0, 0, 0], [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])),
        "nu3": (1280.0, "T2 asymmetric stretch (KEY: C isotope IVPIE)",
                np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])),
        "nu4": (630.0, "T2 asymmetric bend (KEY: F isotope effect)",
                np.array([[0, 0, 0], [0, 1, -1], [0, -1, 1], [1, 0, -1], [-1, 0, 1]])),
    },
    "UF6": {
        "nu3": (620.0, "T1u asymmetric stretch (U isotope IVPIE)",
                np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])),
        "nu4": (180.0, "T1u asymmetric bend",
                np.array([[0, 0, 0], [0, 1, -1], [0, -1, 1], [1, 0, -1], [-1, 0, 1], [0, 0, 0], [0, 0, 0]])),
    },
}


def define_normal_mode_basis(
    molecule: str,
) -> dict[str, np.ndarray]:
    """Get mass-weighted normal mode displacement vectors for a molecule.

    Returns pre-computed unit displacement vectors for each vibrational
    mode. These vectors are the Cartesian displacements corresponding
    to each normal coordinate.

    Args:
        molecule: Molecule identifier: 'BF3', 'CF4', or 'UF6'.

    Returns:
        Dict mapping mode name -> normalized displacement vectors,
        shape (n_atoms, 3) per mode.

    Raises:
        ValueError: If molecule not in the pre-computed mode database.
    """
    if molecule not in MODE_DATA:
        raise ValueError(
            f"Unknown molecule: {molecule}. Available: {list(MODE_DATA.keys())}"
        )

    basis: dict[str, np.ndarray] = {}
    for mode_name, (freq, desc, disp) in MODE_DATA[molecule].items():
        # Normalize displacement vectors
        norm = np.linalg.norm(disp)
        if norm > 1e-15:
            basis[mode_name] = disp / norm
        else:
            basis[mode_name] = disp

    return basis


def project_trajectory_onto_modes(
    trajectory: np.ndarray,
    mode_basis: dict[str, np.ndarray],
    equilibrium_coords: np.ndarray,
) -> dict[str, np.ndarray]:
    """Project each frame onto normal mode displacement vectors.

    Computes the instantaneous mode amplitude q_k(t) = displacement
    from equilibrium projected onto mode k's displacement vector.

    Args:
        trajectory: Centroid positions, shape (n_frames, n_atoms, 3) in Ang.
        mode_basis: Mode displacement vectors from define_normal_mode_basis.
        equilibrium_coords: Equilibrium geometry, shape (n_atoms, 3) in Ang.

    Returns:
        Dict mapping mode name -> amplitude time series, shape (n_frames,).
    """
    n_frames = trajectory.shape[0]

    # Mass weighting (using atomic numbers as proxy for mass weighting)
    # For exact treatment, use actual masses; here we use equal weight
    # since displacement vectors are already mass-weighted in the basis

    amplitudes: dict[str, np.ndarray] = {}

    for mode_name, disp in mode_basis.items():
        # Displacement from equilibrium
        delta = trajectory - equilibrium_coords[np.newaxis, :, :]  # (n_frames, n_atoms, 3)

        # Project: q_k = sum_i delta_i * disp_{k,i}
        # Dot product over atoms and Cartesian directions
        q_k = np.sum(delta * disp[np.newaxis, :, :], axis=(1, 2))  # (n_frames,)
        amplitudes[mode_name] = q_k

    return amplitudes


def compute_mode_specific_free_energy(
    mode_amplitudes: dict[str, np.ndarray],
    delta_a_total: float,
    temperature: float = 145.0,
) -> dict[str, float]:
    """Estimate per-mode contribution to total free energy difference.

    Uses the variance ratio of mode amplitudes as a proxy for the
    mode's contribution to the quantum free energy. Modes with
    larger amplitude fluctuations contribute more to DeltaA.

    Args:
        mode_amplitudes: Mode amplitude time series from project_trajectory_onto_modes.
        delta_a_total: Total DeltaA from FEP calculation (eV).
        temperature: Temperature in K.

    Returns:
        Dict mapping mode name -> estimated DeltaA contribution (eV).
    """
    # Compute variance of each mode
    variances: dict[str, float] = {}
    for mode_name, amp in mode_amplitudes.items():
        variances[mode_name] = float(np.var(amp))

    total_var = sum(variances.values())

    if total_var < 1e-15:
        return {name: 0.0 for name in mode_amplitudes}

    contributions: dict[str, float] = {}
    for mode_name, var in variances.items():
        contributions[mode_name] = round(delta_a_total * var / total_var, 8)

    return contributions


def compute_mode_decomposition_report(
    molecule: str,
    mode_amplitudes: dict[str, np.ndarray],
    mode_contributions: dict[str, float],
    delta_a_total: float,
) -> str:
    """Generate a human-readable mode decomposition report.

    Args:
        molecule: Molecule identifier.
        mode_amplitudes: Mode amplitude time series.
        mode_contributions: Per-mode DeltaA contributions.
        delta_a_total: Total free energy difference (eV).

    Returns:
        Formatted multi-line report string.
    """
    prefix = f"Vibrational Mode Decomposition: {molecule}"

    lines = [
        "=" * 65,
        prefix,
        "=" * 65,
        f"{'Mode':<8} {'Description':<40} {'DeltaA (meV)':<15} {'%':<8}",
        "-" * 65,
    ]

    total_mev = delta_a_total * 1000

    # Sort by contribution magnitude
    sorted_modes = sorted(mode_contributions.items(), key=lambda x: abs(x[1]), reverse=True)

    for mode_name, contribution in sorted_modes:
        _, desc, _ = MODE_DATA.get(molecule, {}).get(
            mode_name, (0.0, mode_name, np.zeros(1))
        )
        contrib_mev = contribution * 1000
        pct = abs(contribution) / max(abs(delta_a_total), 1e-12) * 100

        marker = " <-- DOMINANT" if pct > 40 else ""
        lines.append(
            f"{mode_name:<8} {desc:<40} {contrib_mev:>+8.3f}       {pct:>5.1f}%{marker}"
        )

    lines.append("-" * 65)
    lines.append(f"{'TOTAL':<8} {'':<40} {total_mev:>+8.3f}       {'100.0%':>5}")
    lines.append("=" * 65)

    # Physical interpretation
    if molecule == "BF3":
        dominant_mode = sorted_modes[0][0]
        if dominant_mode == "nu2":
            lines.append(
                "\nInterpretation: The nu2 out-of-plane umbrella bending mode\n"
                "dominates the isotope effect. In the liquid phase, steric\n"
                "hindrance from neighboring molecules causes a blue-shift of nu2,\n"
                "making the ZPE penalty larger for 10B (lighter isotope). This\n"
                "drives inverse vapor pressure isotope effect: 11BF3 is more\n"
                "volatile and enriches at the top of the distillation column."
            )

    return "\n".join(lines)


def plot_mode_contributions(
    mode_amplitudes: dict[str, np.ndarray],
    mode_contributions: dict[str, float],
    output_path: Path,
    molecule: str = "BF3",
) -> None:
    """Generate per-mode amplitude distribution and contribution bar chart.

    Args:
        mode_amplitudes: Mode amplitude time series.
        mode_contributions: Per-mode DeltaA contributions.
        output_path: Path to save PNG figure.
        molecule: Molecule identifier for plot title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_modes = len(mode_amplitudes)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Vibrational Mode Decomposition: {molecule}", fontsize=14, fontweight="bold")

    # Panel 1: amplitude distributions
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (mode_name, amp) in enumerate(mode_amplitudes.items()):
        ax1.hist(amp, bins=50, alpha=0.5, label=mode_name, color=colors[i % len(colors)],
                 density=True)
    ax1.set_xlabel("Mode Amplitude (Ang)")
    ax1.set_ylabel("Density")
    ax1.set_title("Amplitude Distributions")
    ax1.legend(fontsize=8)

    # Panel 2: contribution bar chart
    modes = list(mode_contributions.keys())
    values = [mode_contributions[m] * 1000 for m in modes]
    bar_colors = [colors[i % len(colors)] for i in range(len(modes))]
    bars = ax2.bar(modes, values, color=bar_colors, edgecolor="white")
    ax2.set_ylabel("DeltaA Contribution (meV)")
    ax2.set_title("Per-Mode Free Energy Contribution")
    ax2.axhline(y=0, color="black", linewidth=0.5)

    # Annotate bar values
    for bar, val in zip(bars, values):
        ax2.annotate(f"{val:+.3f}", (bar.get_x() + bar.get_width() / 2, 0),
                     textcoords="offset points", xytext=(0, 5 if val >= 0 else -12),
                     ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Mode decomposition plot saved to: {output_path}")
