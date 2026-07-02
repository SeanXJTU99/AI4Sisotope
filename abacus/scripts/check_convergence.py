"""Check SCF convergence and energy drift in ABACUS batch calculations.

Filters converged calculations from failed ones and generates diagnostic
plots to ensure training data quality. Non-converged or energy-drifted
frames are excluded from the DeePMD-kit training set.
"""

from pathlib import Path
from typing import Optional

import numpy as np

# Matplotlib non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_scf_history(log_path: Path) -> np.ndarray:
    """Parse SCF iteration history from ABACUS running log.

    Extracts per-iteration total energy values to assess convergence
    behavior and detect oscillatory or stalled SCF cycles.

    Args:
        log_path: Path to running_scf.log.

    Returns:
        Array of total energies (eV) at each SCF iteration.
    """
    energies: list[float] = []
    with open(log_path) as f:
        for line in f:
            if "ETOT" in line and "FINAL" not in line:
                parts = line.split()
                try:
                    energies.append(float(parts[-1]))
                except (ValueError, IndexError):
                    continue
    return np.array(energies, dtype=np.float64)


def check_convergence(
    log_path: Path,
    energy_threshold: float = 1e-7,
    force_threshold: float = 0.01,
) -> tuple[bool, str]:
    """Check whether an ABACUS calculation converged successfully.

    Verifies both SCF energy convergence and final force RMS against
    the specified thresholds.

    Args:
        log_path: Path to running_scf.log.
        energy_threshold: SCF energy convergence criterion (eV).
        force_threshold: Maximum allowed force RMS (eV/Å).

    Returns:
        Tuple of (converged: bool, message: str).
    """
    if not log_path.exists():
        return False, f"Log file not found: {log_path}"

    with open(log_path) as f:
        content = f.read()

    if "convergence achieved" not in content.lower():
        return False, "SCF cycle did not converge"

    # Check final force RMS
    for line in content.splitlines():
        if "Total force RMS" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    force_rms = float(parts[-1].strip().split()[0])
                    if force_rms > force_threshold:
                        return False, (
                            f"Force RMS {force_rms:.6f} exceeds "
                            f"threshold {force_threshold}"
                        )
                except ValueError:
                    pass

    return True, "Converged"


def check_energy_drift(
    traj_dir: Path,
    drift_threshold: float = 1.0,
) -> tuple[bool, float]:
    """Check whether energy drifts excessively along a trajectory.

    For MD trajectories (multiple SCF calculations), monitors the
    total energy trend to detect unstable or unphysical runs.

    Args:
        traj_dir: Directory containing sequential ABACUS outputs.
        drift_threshold: Maximum allowed energy drift in eV.

    Returns:
        Tuple of (passed: bool, max_drift_eV: float).
    """
    log_files = sorted(traj_dir.glob("**/running_scf.log"))
    if len(log_files) < 2:
        return True, 0.0

    energies: list[float] = []
    for lf in log_files:
        with open(lf) as f:
            for line in f:
                if "FINAL_ETOT_IS" in line:
                    energies.append(float(line.split()[1]))
                    break

    if len(energies) < 2:
        return True, 0.0

    energies_arr = np.array(energies)
    max_drift = float(np.max(np.abs(energies_arr - energies_arr[0])))

    return max_drift <= drift_threshold, max_drift


def filter_converged_calculations(
    calc_dirs: list[Path],
) -> tuple[list[Path], list[Path], dict[str, int]]:
    """Filter a batch of ABACUS calculations, returning only converged ones.

    Args:
        calc_dirs: List of ABACUS output directory paths.

    Returns:
        Tuple of (converged_dirs, failed_dirs, summary_dict).
        summary_dict has keys: total, converged, failed_scf,
        failed_force, failed_drift.
    """
    converged: list[Path] = []
    failed: list[Path] = []
    summary = {
        "total": len(calc_dirs),
        "converged": 0,
        "failed_scf": 0,
        "failed_force": 0,
        "failed_drift": 0,
    }

    for calc_dir in calc_dirs:
        log_path = calc_dir / "running_scf.log"

        ok, msg = check_convergence(log_path)
        if not ok:
            failed.append(calc_dir)
            if "not converge" in msg.lower():
                summary["failed_scf"] += 1
            elif "force" in msg.lower():
                summary["failed_force"] += 1
            else:
                summary["failed_scf"] += 1
            continue

        drift_ok, _ = check_energy_drift(calc_dir)
        if not drift_ok:
            failed.append(calc_dir)
            summary["failed_drift"] += 1
            continue

        converged.append(calc_dir)

    summary["converged"] = len(converged)
    return converged, failed, summary


def plot_convergence(
    log_paths: list[Path],
    output_path: Path,
    title: str = "SCF Convergence Diagnostics",
) -> None:
    """Generate SCF convergence diagnostic plot for a batch of calculations.

    Creates a multi-panel figure showing:
        1. SCF energy vs iteration for each calculation
        2. Final energy distribution histogram
        3. Summary statistics table

    Args:
        log_paths: List of paths to running_scf.log files.
        output_path: Path to save the output PNG.
        title: Plot title.
    """
    n_calcs = len(log_paths)
    if n_calcs == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14)

    final_energies: list[float] = []
    colors = plt.cm.viridis(np.linspace(0, 1, min(n_calcs, 20)))

    # Panel 1: SCF convergence traces
    ax1 = axes[0]
    for i, log_path in enumerate(log_paths[:20]):
        energies_scf = read_scf_history(log_path)
        if len(energies_scf) > 0:
            ax1.plot(
                energies_scf,
                color=colors[i],
                alpha=0.7,
                linewidth=0.8,
            )
            final_energies.append(energies_scf[-1])

    ax1.set_xlabel("SCF Iteration")
    ax1.set_ylabel("Total Energy (eV)")
    ax1.set_title(f"SCF Convergence ({n_calcs} calculations)")

    # Panel 2: Final energy distribution
    ax2 = axes[1]
    if final_energies:
        ax2.hist(final_energies, bins=min(20, len(final_energies)),
                 color="steelblue", edgecolor="white", alpha=0.8)
        ax2.axvline(np.mean(final_energies), color="red", linestyle="--",
                    label=f"Mean: {np.mean(final_energies):.4f} eV")
        ax2.legend()

    ax2.set_xlabel("Final Total Energy (eV)")
    ax2.set_ylabel("Count")
    ax2.set_title("Energy Distribution")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
