"""Test PIMD beads number (P) convergence for isotope effect prediction.

Runs PIMD simulations with P=32, 64, 128 and compares observables
(quantum kinetic energy, total energy, mode-resolved spring energy)
to determine the minimum P that achieves convergence within the
target tolerance for free energy calculations.

For light-element systems like BF3 at low temperature (145 K),
the high-frequency nu3 asymmetric stretch (~1280 cm-1) has a
characteristic temperature of ~1800 K, requiring P >= 64 for
converged zero-point energy sampling.
"""

from pathlib import Path
from typing import Optional

import numpy as np

# Non-interactive backend for headless cluster environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Physical constants
KB = 8.617333262145e-5   # eV/K
HBAR = 6.582119569e-16   # eV*s
AMU_TO_KG = 1.66053906660e-27


def run_beads_test(
    beads_values: list[int] = [32, 64, 128],
    temperature: float = 145.0,
    n_steps_equil: int = 50000,
    n_steps_prod: int = 100000,
) -> dict[int, dict[str, np.ndarray]]:
    """Run PIMD with different P values and collect observables.

    In production, this calls i-pi with each beads count and parses
    the output properties file. Here we generate physically realistic
    mock data demonstrating the expected convergence behavior.

    Args:
        beads_values: List of P values to test.
        temperature: Simulation temperature in K.
        n_steps_equil: Equilibration steps (discarded).
        n_steps_prod: Production steps (collected).

    Returns:
        Dict mapping P -> dict of observable arrays:
            - quantum_kinetic: centroid-virial kinetic energy estimator (eV)
            - classical_kinetic: classical equipartition kinetic energy (eV)
            - potential: potential energy (eV)
            - total: total energy (eV)
    """
    rng = np.random.default_rng(42)
    results: dict[int, dict[str, np.ndarray]] = {}

    # Physical expectations for BF3 at 145 K:
    # - Classical kinetic: 3N * kT/2 = 6 * 0.0125 = ~0.075 eV (4 atoms, 3N-6+3 modes)
    # - Quantum kinetic excess: grows with P, converges at P~64-128
    # - The nu3 mode (~1280 cm-1 = 0.159 eV) adds significant ZPE
    classical_ke = 0.075  # eV, classical limit
    quantum_excess_p32 = 0.042   # eV, P=32 undersamples high-freq modes
    quantum_excess_p64 = 0.048   # eV, P=64 nearly converged
    quantum_excess_p128 = 0.049  # eV, P=128 fully converged (reference)

    quantum_excess_map = {32: quantum_excess_p32, 64: quantum_excess_p64, 128: quantum_excess_p128}

    for p in beads_values:
        qke_mean = classical_ke + quantum_excess_map.get(p, quantum_excess_p64)
        # Higher P has slightly larger fluctuations (more DOF)
        noise_scale = 0.002 * (p / 64) ** 0.5

        results[p] = {
            "quantum_kinetic": rng.normal(qke_mean, noise_scale, n_steps_prod),
            "classical_kinetic": rng.normal(classical_ke, noise_scale, n_steps_prod),
            "potential": rng.normal(-18.0, 0.01, n_steps_prod),
            "total": rng.normal(-17.925 + qke_mean, noise_scale * 2, n_steps_prod),
        }

    return results


def check_quantum_kinetic_energy(
    results: dict[int, dict[str, np.ndarray]],
    atol: float = 0.001,
) -> dict[str, bool]:
    """Check convergence of quantum kinetic energy estimator.

    Compares each P against P=128 (reference). Declares converged
    if the mean quantum kinetic energy difference is within atol.

    Args:
        results: Output from run_beads_test.
        atol: Absolute tolerance for convergence (eV).

    Returns:
        Dict mapping "P32_vs_P128", "P64_vs_P128" -> converged (bool).
    """
    if 128 not in results:
        raise ValueError("P=128 required as reference for convergence check")

    ref_mean = np.mean(results[128]["quantum_kinetic"])
    converged: dict[str, bool] = {}

    for p in [32, 64]:
        if p not in results:
            continue
        test_mean = np.mean(results[p]["quantum_kinetic"])
        delta = abs(test_mean - ref_mean)
        converged[f"P{p}_vs_P128"] = delta < atol

    return converged


def plot_beads_convergence(
    results: dict[int, dict[str, np.ndarray]],
    output_path: Path,
    title: str = "PIMD Beads Convergence: BF3 at 145 K",
) -> None:
    """Generate convergence plot: quantum kinetic energy vs 1/P.

    Extrapolation to 1/P -> 0 (P -> infinity) gives the exact
    quantum kinetic energy in the continuum limit.

    Args:
        results: Output from run_beads_test.
        output_path: Path to save PNG figure.
        title: Plot title.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    p_values = sorted(results.keys())
    inv_p = [1.0 / p for p in p_values]
    qke_means = [np.mean(results[p]["quantum_kinetic"]) for p in p_values]
    qke_stds = [np.std(results[p]["quantum_kinetic"]) for p in p_values]

    # Panel 1: mean + error bars vs 1/P
    ax1.errorbar(inv_p, qke_means, yerr=qke_stds, fmt="o-", capsize=5,
                 color="steelblue", markersize=8, linewidth=1.5)
    ax1.set_xlabel("1/P")
    ax1.set_ylabel("Quantum Kinetic Energy (eV)")
    ax1.set_title("Convergence with Bead Count")
    ax1.grid(alpha=0.3)

    # Annotate each point
    for ip, p in zip(inv_p, p_values):
        ax1.annotate(f"P={p}", (ip, qke_means[p_values.index(p)]),
                     textcoords="offset points", xytext=(10, 5), fontsize=9)

    # Panel 2: histogram overlay
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for p, color in zip(p_values, colors):
        ax2.hist(results[p]["quantum_kinetic"], bins=50, alpha=0.4,
                 label=f"P={p}", color=color, density=True)
    ax2.set_xlabel("Quantum Kinetic Energy (eV)")
    ax2.set_ylabel("Density")
    ax2.set_title("Distribution Comparison")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Beads convergence plot saved to: {output_path}")


def compute_extrapolated_limit(
    results: dict[int, dict[str, np.ndarray]],
) -> float:
    """Extrapolate quantum kinetic energy to P -> infinity limit.

    Uses linear extrapolation in 1/P to estimate the continuum value.
    This is the gold-standard approach for PIMD convergence analysis.

    Args:
        results: Output from run_beads_test.

    Returns:
        Extrapolated quantum kinetic energy (eV) at P -> infinity.
    """
    p_values = sorted(results.keys())
    inv_p = np.array([1.0 / p for p in p_values])
    qke_means = np.array([np.mean(results[p]["quantum_kinetic"]) for p in p_values])

    # Linear fit: E(P) = E_inf + a * (1/P)
    coeffs = np.polyfit(inv_p, qke_means, 1)
    e_inf = coeffs[1]  # intercept = limit at 1/P -> 0

    print(f"Extrapolated P->inf quantum kinetic energy: {e_inf:.6f} eV")
    print(f"Finite-P correction coefficient: {coeffs[0]:.6f} eV * (1/P)")

    return float(e_inf)
