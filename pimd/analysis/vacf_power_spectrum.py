"""Velocity auto-correlation function (VACF) and vibrational power spectrum.

Computes the VACF from PIMD centroid trajectories and converts it to
a vibrational density of states (power spectrum) via Fourier transform.
This identifies the vibrational modes responsible for isotope effects
by comparing liquid and gas phase spectra.

Key observables:
    - Peak positions (cm-1): identify nu2, nu3, nu4 modes
    - Peak shifts: red-shift (softening) or blue-shift (stiffening) in liquid
    - Integrated intensity: mode contribution to free energy
"""

from pathlib import Path
from typing import Optional

import numpy as np
from scipy import signal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_vacf(
    velocities: np.ndarray,
    max_lag: Optional[int] = None,
) -> np.ndarray:
    """Compute normalized velocity auto-correlation function.

    C(tau) = <v(0) * v(tau)> / <v(0) * v(0)>

    Uses FFT-based convolution for O(N log N) computation.

    Args:
        velocities: Atomic velocities, shape (n_frames, n_atoms, 3) in Ang/fs.
        max_lag: Maximum time lag in frames. Default: n_frames // 2.

    Returns:
        VACF array of shape (max_lag,).
    """
    n_frames, n_atoms, _ = velocities.shape
    if max_lag is None:
        max_lag = n_frames // 2

    vacf = np.zeros(max_lag)

    for atom in range(n_atoms):
        for dim in range(3):
            v = velocities[:, atom, dim]
            v = v - np.mean(v)  # remove drift
            # Zero-padded FFT convolution
            n_fft = 2 ** int(np.ceil(np.log2(2 * n_frames - 1)))
            v_fft = np.fft.rfft(v, n=n_fft)
            corr = np.fft.irfft(v_fft * np.conj(v_fft), n=n_fft)[:max_lag]
            vacf += corr

    # Normalize
    if vacf[0] > 1e-15:
        vacf /= vacf[0]
    else:
        vacf = np.zeros_like(vacf)

    return vacf


def compute_power_spectrum(
    vacf: np.ndarray,
    timestep: float,
    pad_factor: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute vibrational power spectrum via FFT of VACF.

    Args:
        vacf: VACF from compute_vacf(), shape (max_lag,).
        timestep: MD timestep in femtoseconds.
        pad_factor: Zero-padding factor for frequency resolution.

    Returns:
        Tuple of (frequencies_cm1, power_spectrum).
        frequencies: shape (n_freqs,) in wavenumbers (cm^-1).
        power_spectrum: shape (n_freqs,), normalized to max=1.
    """
    n = len(vacf) * pad_factor

    # FFT of VACF -> power spectrum
    # Multiply by cos window to reduce ringing from truncation
    window = np.cos(np.linspace(0, np.pi / 2, len(vacf)))
    vacf_windowed = vacf * window

    spectrum_raw = np.abs(np.fft.rfft(vacf_windowed, n=n))
    spectrum_raw /= np.max(spectrum_raw) if np.max(spectrum_raw) > 0 else 1.0

    # Frequency axis: fs -> cm^-1
    # 1 cm^-1 = 1 / (c * lambda) where c = 2.9979e10 cm/s
    # For time-series with dt in fs: f(cm^-1) = f(frame) / (N * dt * c * 1e-13)
    C_LIGHT = 2.99792458e10  # cm/s
    dt_s = timestep * 1e-15  # fs -> s
    freqs = np.fft.rfftfreq(n, d=dt_s)
    freqs_cm1 = freqs / (C_LIGHT * 1e-2)  # Hz -> cm^-1

    return freqs_cm1[:len(spectrum_raw)], spectrum_raw


def identify_peak_frequencies(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    height: float = 0.01,
    min_distance: float = 20.0,
) -> dict[str, np.ndarray]:
    """Identify vibrational mode peaks in power spectrum.

    Uses scipy.signal.find_peaks with physically motivated constraints.

    Args:
        freqs: Frequency array in cm^-1.
        spectrum: Power spectrum array.
        height: Minimum peak height (fraction of max).
        min_distance: Minimum separation between peaks in cm^-1.

    Returns:
        Dict with keys 'peak_indices', 'peak_freqs', 'peak_heights'.
    """
    max_val = np.max(spectrum)
    if max_val < 1e-15:
        return {"peak_indices": np.array([]), "peak_freqs": np.array([]),
                "peak_heights": np.array([])}

    peak_indices, properties = signal.find_peaks(
        spectrum,
        height=height * max_val,
        distance=int(min_distance / (freqs[1] - freqs[0])) if len(freqs) > 1 else 1,
    )

    return {
        "peak_indices": peak_indices,
        "peak_freqs": freqs[peak_indices],
        "peak_heights": spectrum[peak_indices] if len(peak_indices) > 0 else np.array([]),
    }


def compare_gas_liquid_spectra(
    freqs_gas: np.ndarray,
    spectrum_gas: np.ndarray,
    freqs_liquid: np.ndarray,
    spectrum_liquid: np.ndarray,
    mode_labels: Optional[dict[float, str]] = None,
) -> dict[str, dict]:
    """Compare gas and liquid phase power spectra.

    Identifies frequency shifts (red-shift = softening in liquid,
    blue-shift = stiffening in liquid) for each mode, which reveals
    the physical mechanism driving the isotope effect.

    Args:
        freqs_gas: Gas phase frequencies (cm^-1).
        spectrum_gas: Gas phase power spectrum.
        freqs_liquid: Liquid phase frequencies (cm^-1).
        spectrum_liquid: Liquid phase power spectrum.
        mode_labels: Optional mapping from reference frequency to mode name.

    Returns:
        Dict mapping mode name -> {gas_peak, liquid_peak, shift_cm1, interpretation}.
    """
    peaks_gas = identify_peak_frequencies(freqs_gas, spectrum_gas)
    peaks_liquid = identify_peak_frequencies(freqs_liquid, spectrum_liquid)

    if mode_labels is None:
        # Default for BF3: nu1 (~888), nu2 (~691), nu3 (~1503), nu4 (~480)
        mode_labels = {
            480.0: "nu4 (in-plane bend)",
            691.0: "nu2 (out-of-plane bend)",
            888.0: "nu1 (symmetric stretch)",
            1503.0: "nu3 (asymmetric stretch)",
        }

    comparison: dict[str, dict] = {}

    for ref_freq, mode_name in mode_labels.items():
        # Find nearest peak in each phase
        gas_match = _find_nearest_peak(peaks_gas, ref_freq)
        liquid_match = _find_nearest_peak(peaks_liquid, ref_freq)

        if gas_match and liquid_match:
            shift = liquid_match["freq"] - gas_match["freq"]
            if shift > 5.0:
                interp = "blue-shift (stiffening in liquid)"
            elif shift < -5.0:
                interp = "red-shift (softening in liquid)"
            else:
                interp = "no significant shift"

            comparison[mode_name] = {
                "gas_freq_cm1": round(gas_match["freq"], 1),
                "liquid_freq_cm1": round(liquid_match["freq"], 1),
                "shift_cm1": round(shift, 1),
                "interpretation": interp,
            }

    return comparison


def plot_power_spectrum(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    peaks: dict[str, np.ndarray],
    output_path: Path,
    mode_labels: Optional[dict[float, str]] = None,
    title: str = "Vibrational Power Spectrum",
) -> None:
    """Plot power spectrum with labeled mode peaks.

    Args:
        freqs: Frequency array (cm^-1).
        spectrum: Power spectrum array.
        peaks: Peak identification from identify_peak_frequencies.
        output_path: Path to save PNG.
        mode_labels: Optional mode label mapping.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(freqs, spectrum, color="steelblue", linewidth=1.0, alpha=0.8)
    ax.fill_between(freqs, 0, spectrum, color="steelblue", alpha=0.15)

    # Mark peaks
    if len(peaks["peak_freqs"]) > 0:
        ax.scatter(peaks["peak_freqs"], peaks["peak_heights"],
                   color="red", s=60, zorder=5, marker="^")

        # Label peaks using mode_labels
        if mode_labels:
            for pf, ph in zip(peaks["peak_freqs"], peaks["peak_heights"]):
                for ref_freq, label in mode_labels.items():
                    if abs(pf - ref_freq) < 50:
                        ax.annotate(label, (pf, ph),
                                    textcoords="offset points",
                                    xytext=(0, 12), fontsize=9,
                                    ha="center", color="darkred")
                        break

    ax.set_xlabel("Frequency (cm$^{-1}$)")
    ax.set_ylabel("Intensity (arb. units)")
    ax.set_title(title)
    ax.set_xlim(0, 2000)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Power spectrum plot saved to: {output_path}")


def _find_nearest_peak(
    peaks: dict[str, np.ndarray],
    target_freq: float,
    tolerance: float = 100.0,
) -> Optional[dict[str, float]]:
    """Find the spectral peak nearest to a target frequency."""
    if len(peaks["peak_freqs"]) == 0:
        return None

    distances = np.abs(peaks["peak_freqs"] - target_freq)
    min_idx = np.argmin(distances)

    if distances[min_idx] < tolerance:
        return {
            "freq": float(peaks["peak_freqs"][min_idx]),
            "height": float(peaks["peak_heights"][min_idx]),
        }
    return None
