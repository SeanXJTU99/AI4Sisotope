"""Plot DeePMD-kit training learning curves from lcurve.out.

Parses the training log to extract energy and force RMSE evolution
over training steps, and generates publication-quality plots.
"""

from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_lcurve(lcurve_path: Path) -> dict[str, np.ndarray]:
    """Parse a DeePMD-kit lcurve.out file into structured arrays.

    The lcurve.out file contains space-separated columns:
        step  rmse_e_tr  rmse_e_va  rmse_f_tr  rmse_f_va
    with a header line starting with '#'.

    Args:
        lcurve_path: Path to lcurve.out file.

    Returns:
        Dict with keys: 'step', 'rmse_e_tr', 'rmse_e_va',
        'rmse_f_tr', 'rmse_f_va'. All arrays shape (n_steps,).

    Raises:
        FileNotFoundError: If lcurve.out does not exist.
    """
    if not lcurve_path.exists():
        raise FileNotFoundError(f"Learning curve file not found: {lcurve_path}")

    data: list[list[float]] = []
    with open(lcurve_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                data.append([float(x) for x in parts[:5]])

    arr = np.array(data)
    return {
        "step": arr[:, 0],
        "rmse_e_tr": arr[:, 1],
        "rmse_e_va": arr[:, 2] if arr.shape[1] > 2 else arr[:, 1],
        "rmse_f_tr": arr[:, 3] if arr.shape[1] > 3 else arr[:, 2],
        "rmse_f_va": arr[:, 4] if arr.shape[1] > 4 else arr[:, 3] if arr.shape[1] > 3 else arr[:, 2],
    }


def plot_energy_rmse(
    steps: np.ndarray,
    rmse_tr: np.ndarray,
    rmse_va: np.ndarray,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot energy RMSE learning curve on given axes.

    Args:
        steps: Training step numbers.
        rmse_tr: Training energy RMSE.
        rmse_va: Validation energy RMSE.
        ax: Optional matplotlib Axes to plot on.

    Returns:
        The matplotlib Axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.plot(steps, rmse_tr, alpha=0.6, linewidth=0.8, label="Training", color="#1f77b4")
    ax.plot(steps, rmse_va, alpha=0.9, linewidth=1.2, label="Validation", color="#d62728")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Energy RMSE (eV)")
    ax.set_title("Energy Learning Curve")
    ax.legend()
    ax.grid(alpha=0.3)

    return ax


def plot_force_rmse(
    steps: np.ndarray,
    rmse_tr: np.ndarray,
    rmse_va: np.ndarray,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot force RMSE learning curve on given axes.

    Args:
        steps: Training step numbers.
        rmse_tr: Training force RMSE (eV/Ang).
        rmse_va: Validation force RMSE (eV/Ang).
        ax: Optional matplotlib Axes to plot on.

    Returns:
        The matplotlib Axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.plot(steps, rmse_tr, alpha=0.6, linewidth=0.8, label="Training", color="#1f77b4")
    ax.plot(steps, rmse_va, alpha=0.9, linewidth=1.2, label="Validation", color="#d62728")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Force RMSE (eV/Ang)")
    ax.set_title("Force Learning Curve")
    ax.legend()
    ax.grid(alpha=0.3)

    return ax


def plot_combined(
    lcurve_data: dict[str, np.ndarray],
    output_path: Path,
    title: str = "DPA-2 Training: BF3 Isotope Effect Force Field",
) -> None:
    """Generate a combined energy + force subplot figure.

    Args:
        lcurve_data: Output from read_lcurve().
        output_path: Path to save the PNG figure.
        title: Overall figure title.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    plot_energy_rmse(
        lcurve_data["step"],
        lcurve_data["rmse_e_tr"],
        lcurve_data["rmse_e_va"],
        ax=ax1,
    )

    plot_force_rmse(
        lcurve_data["step"],
        lcurve_data["rmse_f_tr"],
        lcurve_data["rmse_f_va"],
        ax=ax2,
    )

    # Annotate final values
    ax1.annotate(
        f"Final: {lcurve_data['rmse_e_va'][-1]:.4f} eV",
        xy=(lcurve_data["step"][-1], lcurve_data["rmse_e_va"][-1]),
        xytext=(0.6, 0.9),
        textcoords="axes fraction",
        fontsize=9,
        color="#d62728",
    )
    ax2.annotate(
        f"Final: {lcurve_data['rmse_f_va'][-1]:.4f} eV/Ang",
        xy=(lcurve_data["step"][-1], lcurve_data["rmse_f_va"][-1]),
        xytext=(0.6, 0.9),
        textcoords="axes fraction",
        fontsize=9,
        color="#d62728",
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Learning curve plot saved to: {output_path}")


def generate_mock_lcurve(output_path: Path) -> dict[str, np.ndarray]:
    """Generate a mock lcurve.out for demonstration purposes.

    Simulates realistic DPA-2 training convergence behavior for BF3.

    Args:
        output_path: Where to write the mock lcurve.out file.

    Returns:
        Dict with the mock lcurve data.
    """
    rng = np.random.default_rng(42)
    n_steps = 1000
    steps = np.arange(1000, 1000 * (n_steps + 1), 1000)

    # Realistic convergence: exponential decay + noise
    def decay(step: np.ndarray, start: float, end: float) -> np.ndarray:
        return end + (start - end) * np.exp(-step / 200000)

    rmse_e_tr_base = decay(steps, 0.05, 0.002)
    rmse_e_va_base = decay(steps, 0.08, 0.003)
    rmse_f_tr_base = decay(steps, 0.15, 0.02)
    rmse_f_va_base = decay(steps, 0.20, 0.03)

    rmse_e_tr = rmse_e_tr_base + rng.normal(0, 0.0002, n_steps)
    rmse_e_va = rmse_e_va_base + rng.normal(0, 0.0003, n_steps)
    rmse_f_tr = rmse_f_tr_base + rng.normal(0, 0.001, n_steps)
    rmse_f_va = rmse_f_va_base + rng.normal(0, 0.002, n_steps)

    with open(output_path, "w") as f:
        f.write("# step rmse_e_tr rmse_e_va rmse_f_tr rmse_f_va\n")
        for i in range(n_steps):
            f.write(f"{int(steps[i])} {rmse_e_tr[i]:.6e} {rmse_e_va[i]:.6e} "
                    f"{rmse_f_tr[i]:.6e} {rmse_f_va[i]:.6e}\n")

    return {
        "step": steps,
        "rmse_e_tr": rmse_e_tr,
        "rmse_e_va": rmse_e_va,
        "rmse_f_tr": rmse_f_tr,
        "rmse_f_va": rmse_f_va,
    }
