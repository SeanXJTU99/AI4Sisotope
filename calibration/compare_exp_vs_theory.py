"""Compare theoretical alpha predictions against industrial experimental data.

Generates side-by-side comparison tables and publication-quality plots
showing agreement between PIMD-FEP predictions and distillation column
measurements across multiple isotope systems.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class ComparisonResult:
    """Side-by-side comparison of one data point."""
    temperature: float
    alpha_exp: float
    alpha_exp_std: float
    alpha_theory: float
    alpha_theory_std: float
    deviation_sigma: float  # |theory - exp| / sqrt(sigma_exp^2 + sigma_theory^2)
    isotope_pair: str


def load_mock_experimental_data(molecule: str) -> dict[str, np.ndarray]:
    """Load mock experimental alpha data for a given molecule.

    Simulates industrial distillation column measurements at multiple
    temperature points around the boiling point.

    Args:
        molecule: Molecule identifier ('BF3', 'CF4', or 'UF6').

    Returns:
        Dict with keys: 'temperature', 'alpha', 'alpha_std'.
    """
    rng = np.random.default_rng(hash(molecule) % (2**31))

    config = {
        "BF3": {"T_range": (110, 210), "base_alpha": 1.0045, "T_ref": 145.0},
        "CF4": {"T_range": (100, 200), "base_alpha": 1.0030, "T_ref": 130.0},
        "UF6": {"T_range": (300, 400), "base_alpha": 1.0015, "T_ref": 330.0},
    }

    cfg = config.get(molecule, config["BF3"])
    n_points = 8
    temperatures = np.linspace(cfg["T_range"][0], cfg["T_range"][1], n_points)
    alpha = 1.0 + (cfg["base_alpha"] - 1.0) * np.exp(
        -(temperatures - cfg["T_range"][0]) / (cfg["T_ref"] * 0.5)
    )
    alpha += rng.normal(0, 0.0003, n_points)

    return {
        "temperature": temperatures,
        "alpha": alpha,
        "alpha_std": np.full(n_points, 0.0005),
    }


def compute_agreement_metrics(
    alpha_theory: np.ndarray,
    alpha_exp: np.ndarray,
    alpha_exp_std: np.ndarray,
) -> dict[str, float]:
    """Compute quantitative agreement metrics.

    Args:
        alpha_theory: Predicted alpha values.
        alpha_exp: Experimental alpha values.
        alpha_exp_std: Experimental uncertainties.

    Returns:
        Dict with MAE, RMSE, R-squared, and fraction within 1-sigma/2-sigma.
    """
    diff = alpha_theory - alpha_exp
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((alpha_exp - np.mean(alpha_exp)) ** 2)
    r2 = float(1 - ss_res / max(ss_tot, 1e-15))

    # Fraction within confidence intervals
    within_1sigma = np.sum(np.abs(diff) < alpha_exp_std) / len(diff)
    within_2sigma = np.sum(np.abs(diff) < 2 * alpha_exp_std) / len(diff)

    return {
        "MAE": round(mae, 6),
        "RMSE": round(rmse, 6),
        "R2": round(r2, 4),
        "fraction_within_1sigma": round(float(within_1sigma), 3),
        "fraction_within_2sigma": round(float(within_2sigma), 3),
    }


def build_comparison_table(
    exp_data: dict[str, np.ndarray],
    theory_alpha: np.ndarray,
    theory_std: np.ndarray,
    isotope_pair: str,
) -> list[ComparisonResult]:
    """Build a row-by-row comparison table.

    Args:
        exp_data: Experimental data dict from load_mock_experimental_data.
        theory_alpha: Theoretical alpha predictions.
        theory_std: Theoretical alpha uncertainties.
        isotope_pair: Isotope pair identifier.

    Returns:
        List of ComparisonResult objects.
    """
    results: list[ComparisonResult] = []
    n = len(exp_data["temperature"])

    for i in range(n):
        combined_std = np.sqrt(exp_data["alpha_std"][i]**2 + theory_std[i]**2)
        deviation = abs(theory_alpha[i] - exp_data["alpha"][i]) / max(combined_std, 1e-15)
        results.append(ComparisonResult(
            temperature=float(exp_data["temperature"][i]),
            alpha_exp=float(exp_data["alpha"][i]),
            alpha_exp_std=float(exp_data["alpha_std"][i]),
            alpha_theory=float(theory_alpha[i]),
            alpha_theory_std=float(theory_std[i]),
            deviation_sigma=round(float(deviation), 2),
            isotope_pair=isotope_pair,
        ))
    return results


def plot_exp_vs_theory(
    exp_data: dict[str, np.ndarray],
    theory_alpha: np.ndarray,
    theory_std: np.ndarray,
    molecule: str,
    isotope_pair: str,
    output_path: Path,
) -> None:
    """Generate experimental vs theoretical comparison plot.

    Args:
        exp_data: Experimental data dict.
        theory_alpha: Theoretical alpha predictions.
        theory_std: Theoretical alpha uncertainties.
        molecule: Molecule identifier.
        isotope_pair: Isotope pair label.
        output_path: Path to save PNG.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Experiment vs Theory: {molecule} ({isotope_pair})",
                 fontsize=14, fontweight="bold")

    T = exp_data["temperature"]
    alpha_exp = exp_data["alpha"]
    alpha_std = exp_data["alpha_std"]

    # Panel 1: alpha vs temperature
    ax1.errorbar(T, alpha_exp, yerr=alpha_std, fmt="o", color="steelblue",
                 capsize=3, label="Experiment", markersize=6)
    ax1.fill_between(T, theory_alpha - theory_std, theory_alpha + theory_std,
                     alpha=0.2, color="red", label="Theory +/- 1sigma")
    ax1.plot(T, theory_alpha, "r-", linewidth=1.5, label="Theory (calibrated)")
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="alpha=1 (no effect)")
    ax1.set_xlabel("Temperature (K)")
    ax1.set_ylabel("Fractionation Factor alpha")
    ax1.set_title("Alpha vs Temperature")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Panel 2: residuals
    residuals = (alpha_exp - theory_alpha) * 1000  # in 1e-3 units
    ax2.errorbar(T, residuals, yerr=alpha_std * 1000, fmt="o",
                 color="steelblue", capsize=3, markersize=6)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.fill_between(T, -1, 1, alpha=0.1, color="green", label="+/- 1e-3 alpha")
    ax2.set_xlabel("Temperature (K)")
    ax2.set_ylabel("Residual (exp - theory) x 1e-3")
    ax2.set_title("Residuals")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plot saved to: {output_path}")


def generate_calibration_report(
    comparison: list[ComparisonResult],
    metrics: dict[str, float],
    output_path: Path,
) -> None:
    """Generate a Markdown calibration report.

    Args:
        comparison: List of comparison results.
        metrics: Agreement metrics from compute_agreement_metrics.
        output_path: Path to write Markdown report.
    """
    with open(output_path, "w") as f:
        f.write("# Experimental Calibration Report\n\n")
        f.write("## Agreement Metrics\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        for key, val in metrics.items():
            f.write(f"| {key} | {val} |\n")

        f.write("\n## Per-Point Comparison\n\n")
        f.write("| T (K) | alpha_exp | alpha_theory | deviation (sigma) |\n")
        f.write("|-------|-----------|-------------|-------------------|\n")
        for r in comparison:
            f.write(f"| {r.temperature:.1f} | {r.alpha_exp:.6f} +/- {r.alpha_exp_std:.6f} "
                    f"| {r.alpha_theory:.6f} +/- {r.alpha_theory_std:.6f} "
                    f"| {r.deviation_sigma:.1f} |\n")

        f.write("\n## Interpretation\n\n")
        if metrics["fraction_within_2sigma"] >= 0.95:
            f.write("Theory and experiment agree within 2-sigma for >= 95% of points. "
                    "Model is calibrated for industrial process window optimization.\n")
        else:
            f.write("Some points exceed 2-sigma deviation. Consider additional "
                    "PIMD sampling or functional refinement.\n")

    print(f"Calibration report written to: {output_path}")
