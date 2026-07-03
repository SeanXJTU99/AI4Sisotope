"""Calculate isotope fractionation factor alpha from FEP free energies.

Core formula:
    ln(alpha) = -(DeltaA_liquid - DeltaA_gas) / (k_B * T)

Interpretation:
    alpha > 1 -> heavy isotope enriched in gas phase (inverse effect, top of column)
    alpha < 1 -> light isotope enriched in gas phase (normal effect, bottom of column)
    alpha = 1 -> no isotope effect (crossover point)
"""

from pathlib import Path
from typing import Optional

import numpy as np


KB = 8.617333262145e-5  # eV/K


def calculate_alpha(
    delta_a_liquid: float,
    delta_a_gas: float,
    temperature: float,
) -> tuple[float, float]:
    """Calculate isotope fractionation factor alpha.

    Args:
        delta_a_liquid: Free energy difference in liquid phase (eV).
        delta_a_gas: Free energy difference in gas phase (eV).
        temperature: Temperature in K.

    Returns:
        Tuple of (alpha, ln_alpha).
    """
    kt = KB * temperature
    delta_delta_a = delta_a_liquid - delta_a_gas
    ln_alpha = -delta_delta_a / kt
    alpha = np.exp(ln_alpha)

    return float(alpha), float(ln_alpha)


def calculate_alpha_with_uncertainty(
    delta_a_liquid: float,
    delta_a_liquid_std: float,
    delta_a_gas: float,
    delta_a_gas_std: float,
    temperature: float,
) -> dict[str, float]:
    """Calculate alpha with full error propagation.

    Propagates uncertainties from liquid and gas phase FEP calculations
    through the alpha formula using standard error propagation.

    Args:
        delta_a_liquid: DeltaA for liquid phase (eV).
        delta_a_liquid_std: Standard deviation of DeltaA_liquid (eV).
        delta_a_gas: DeltaA for gas phase (eV).
        delta_a_gas_std: Standard deviation of DeltaA_gas (eV).
        temperature: Temperature in K.

    Returns:
        Dict with alpha, ln_alpha, and their uncertainties.
    """
    kt = KB * temperature
    delta_delta_a = delta_a_liquid - delta_a_gas
    delta_delta_a_std = np.sqrt(delta_a_liquid_std**2 + delta_a_gas_std**2)

    ln_alpha = -delta_delta_a / kt
    ln_alpha_std = delta_delta_a_std / kt

    alpha = np.exp(ln_alpha)
    # Propagate: sigma_alpha = alpha * sigma_ln_alpha
    alpha_std = alpha * ln_alpha_std

    return {
        "alpha": round(alpha, 8),
        "alpha_std": round(alpha_std, 8),
        "ln_alpha": round(ln_alpha, 8),
        "ln_alpha_std": round(ln_alpha_std, 8),
        "delta_delta_A_meV": round(delta_delta_a * 1000, 5),
        "delta_delta_A_std_meV": round(delta_delta_a_std * 1000, 5),
        "temperature_K": temperature,
    }


def classify_isotope_effect(
    alpha: float,
    alpha_std: float,
    confidence_sigma: float = 2.0,
) -> str:
    """Classify the isotope effect based on alpha value and confidence.

    Args:
        alpha: Fractionation factor.
        alpha_std: Uncertainty in alpha.
        confidence_sigma: Number of standard deviations for confidence.

    Returns:
        One of: 'inverse', 'normal', 'crossover', 'uncertain'.
    """
    n_sigma = abs(alpha - 1.0) / max(alpha_std, 1e-12)

    if n_sigma < confidence_sigma:
        return "uncertain"
    if alpha > 1.0:
        return "inverse"
    elif alpha < 1.0:
        return "normal"
    else:
        return "crossover"


def format_alpha_report(
    alpha: float,
    alpha_std: float,
    temperature: float,
    isotope_pair: str,
) -> str:
    """Format alpha result as a human-readable report.

    Args:
        alpha: Fractionation factor.
        alpha_std: Uncertainty in alpha.
        temperature: Temperature in K.
        isotope_pair: Description of the isotope pair, e.g. '10B/11B'.

    Returns:
        Multi-line formatted report string.
    """
    classification = classify_isotope_effect(alpha, alpha_std)
    ln_alpha = np.log(alpha)

    lines = [
        "=" * 60,
        f"Isotope Fractionation Report",
        f"  Isotope pair:  {isotope_pair}",
        f"  Temperature:   {temperature:.1f} K",
        f"  alpha:         {alpha:.6f} +/- {alpha_std:.6f}",
        f"  ln(alpha):     {ln_alpha:.6f}",
        f"  Classification: {classification.upper()}",
        "=" * 60,
    ]

    if classification == "inverse":
        lines.append(
            "Heavy isotope enriched in gas phase (top of distillation column).\n"
            "Consistent with inverse vapor pressure isotope effect (IVPIE)."
        )
    elif classification == "normal":
        lines.append(
            "Light isotope enriched in gas phase (bottom of column).\n"
            "Consistent with normal vapor pressure isotope effect."
        )
    elif classification == "uncertain":
        lines.append(
            "Cannot distinguish from alpha=1 within confidence bounds.\n"
            "Consider longer PIMD trajectories or higher P for convergence."
        )
    else:
        lines.append("Crossover point -- no net isotope separation.")

    return "\n".join(lines)


def generate_temperature_scan(
    delta_a_liquid: float,
    delta_a_gas: float,
    t_min: float = 80.0,
    t_max: float = 300.0,
    n_points: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate alpha vs temperature curve for a given DeltaDeltaA.

    Useful for identifying crossover temperatures (alpha=1) and
    optimal operating windows for distillation column design.

    Args:
        delta_a_liquid: DeltaA for liquid phase (eV) at reference T.
        delta_a_gas: DeltaA for gas phase (eV) at reference T.
        t_min: Minimum temperature (K).
        t_max: Maximum temperature (K).
        n_points: Number of temperature grid points.

    Returns:
        Tuple of (temperatures, alpha_values).
    """
    temperatures = np.linspace(t_min, t_max, n_points)
    alpha_values = np.zeros(n_points)

    # Note: DeltaA itself has temperature dependence through the
    # spring constant kappa ~ T^2. This simplified scan assumes
    # DeltaDeltaA is approximately constant over the T range,
    # which is reasonable for narrow ranges near the boiling point.
    for i, t in enumerate(temperatures):
        alpha, _ = calculate_alpha(delta_a_liquid, delta_a_gas, t)
        alpha_values[i] = alpha

    return temperatures, alpha_values


def find_crossover_temperature(
    delta_a_liquid: float,
    delta_a_gas: float,
    t_low: float = 50.0,
    t_high: float = 500.0,
) -> Optional[float]:
    """Find the crossover temperature where alpha = 1.

    At the crossover point, normal and inverse isotope effects switch,
    and no net separation occurs. The distillation column cannot
    operate near this temperature.

    For this simplified model, DeltaDeltaA is constant so there
    is a single crossover at T = DeltaDeltaA / 0 = infinity if
    DeltaDeltaA != 0. In practice, DeltaA itself has T-dependence
    through the spring constant, leading to real crossovers.

    Args:
        delta_a_liquid: DeltaA for liquid phase (eV).
        delta_a_gas: DeltaA for gas phase (eV).
        t_low: Lower bound for search (K).
        t_high: Upper bound for search (K).

    Returns:
        Crossover temperature in K, or None if no crossover in range.
    """
    delta_delta_a = delta_a_liquid - delta_a_gas

    # If DeltaDeltaA is the same sign across the range, no crossover
    # Real crossover requires T-dependent DeltaA (beyond this simplified model)
    if abs(delta_delta_a) < 1e-9:
        return None  # always alpha ~ 1

    # Placeholder: in a full implementation, this would solve
    # DeltaA(T) = 0 using the T-dependent spring constant.
    # For now, return None to indicate full model needed.
    return None
