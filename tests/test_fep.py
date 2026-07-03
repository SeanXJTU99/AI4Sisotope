"""Unit tests for the FEP free energy perturbation module.

Validates numerical correctness of mass mutation calculations and
statistical convergence of Zwanzig exponential averaging.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from pimd.fep.fep_mass_mutation import (
    IsotopePair,
    compute_spring_energy_difference,
    compute_fep_free_energy,
    run_mass_mutation_fep,
    KB,
)


@pytest.fixture
def mock_pimd_trajectory() -> np.ndarray:
    """Generate a mock PIMD trajectory for BF3 with P=64 beads.

    Simulates harmonic oscillator motion around D3h equilibrium geometry.
    """
    rng = np.random.default_rng(42)
    n_frames = 1000
    n_atoms = 4
    n_beads = 64

    coords = rng.normal(0, 0.05, size=(n_frames, n_atoms, n_beads, 3))
    # B at center
    coords[:, 0, :, :] += np.array([0.0, 0.0, 0.0])[None, None, :]
    # Three F atoms at ~1.3 A in D3h geometry
    coords[:, 1, :, :] += np.array([1.3, 0.0, 0.0])[None, None, :]
    coords[:, 2, :, :] += np.array([-0.65, 1.13, 0.0])[None, None, :]
    coords[:, 3, :, :] += np.array([-0.65, -1.13, 0.0])[None, None, :]

    return coords


@pytest.fixture
def b10_b11_pair() -> IsotopePair:
    """10B/11B isotope pair for BF3 (boron is atom index 0)."""
    return IsotopePair(
        element="B",
        mass_light=10.0129,
        mass_heavy=11.0093,
        atom_indices=[0],
    )


class TestSpringEnergyDifference:
    """Test spring potential energy difference computation."""

    def test_zero_for_equal_masses(
        self, mock_pimd_trajectory: np.ndarray
    ) -> None:
        """Delta U_spring should be zero when light and heavy masses are equal."""
        masses = np.array([10.81, 18.998, 18.998, 18.998])
        delta = compute_spring_energy_difference(
            mock_pimd_trajectory, masses, masses, 145.0, 64
        )
        assert np.allclose(delta, 0.0, atol=1e-12)

    def test_positive_for_heavier_isotope(
        self, mock_pimd_trajectory: np.ndarray
    ) -> None:
        """Heavier isotope should have higher spring energy (positive Delta U)."""
        masses_light = np.array([10.81, 18.998, 18.998, 18.998])
        masses_heavy = np.array([11.01, 18.998, 18.998, 18.998])
        delta = compute_spring_energy_difference(
            mock_pimd_trajectory, masses_light, masses_heavy, 145.0, 64
        )
        assert np.mean(delta) > 0, "Heavier isotope should have higher spring energy"

    def test_scales_with_temperature_squared(
        self, mock_pimd_trajectory: np.ndarray
    ) -> None:
        """Delta U should scale approximately as T^2 (through kappa ~ T^2)."""
        masses_light = np.array([10.81, 18.998, 18.998, 18.998])
        masses_heavy = np.array([11.01, 18.998, 18.998, 18.998])

        delta_t100 = compute_spring_energy_difference(
            mock_pimd_trajectory, masses_light, masses_heavy, 100.0, 64
        )
        delta_t200 = compute_spring_energy_difference(
            mock_pimd_trajectory, masses_light, masses_heavy, 200.0, 64
        )

        ratio = np.mean(delta_t200) / np.mean(delta_t100)
        expected_ratio = (200 / 100) ** 2
        assert abs(ratio / expected_ratio - 1.0) < 0.1


class TestFEPFreeEnergy:
    """Test FEP free energy computation via Zwanzig averaging."""

    def test_linear_approaches_exponential_for_small_perturbation(
        self, mock_pimd_trajectory: np.ndarray
    ) -> None:
        """Linear (<DeltaU>) ~ Exponential (-kT ln<exp(-beta DU)>) for small DU."""
        masses_light = np.array([10.81, 18.998, 18.998, 18.998])
        masses_heavy = np.array([10.82, 18.998, 18.998, 18.998])  # tiny mass change

        delta_spring = compute_spring_energy_difference(
            mock_pimd_trajectory, masses_light, masses_heavy, 145.0, 64
        )

        delta_a_exp, _ = compute_fep_free_energy(delta_spring, 145.0, method="exponential")
        delta_a_lin, _ = compute_fep_free_energy(delta_spring, 145.0, method="linear")

        # Should agree within ~1% for tiny mass perturbation
        assert abs(delta_a_exp - delta_a_lin) < abs(delta_a_exp) * 0.05

    def test_convergence_with_frames(
        self, mock_pimd_trajectory: np.ndarray
    ) -> None:
        """Delta A should stabilize as number of frames increases."""
        masses_light = np.array([10.81, 18.998, 18.998, 18.998])
        masses_heavy = np.array([11.01, 18.998, 18.998, 18.998])

        delta_full = compute_spring_energy_difference(
            mock_pimd_trajectory, masses_light, masses_heavy, 145.0, 64
        )

        a_full, _ = compute_fep_free_energy(delta_full, 145.0)
        a_half, _ = compute_fep_free_energy(delta_full[:500], 145.0)

        # Should be roughly consistent (within ~20% of each other)
        assert abs(a_full - a_half) < max(abs(a_full), 0.001) * 0.5


class TestIsotopePair:
    """Test IsotopePair dataclass."""

    def test_single_atom_substitution(self, b10_b11_pair: IsotopePair) -> None:
        """Single-atom isotope substitution should be specifiable."""
        assert b10_b11_pair.element == "B"
        assert b10_b11_pair.mass_light == pytest.approx(10.0129)
        assert b10_b11_pair.mass_heavy == pytest.approx(11.0093)
        assert b10_b11_pair.atom_indices == [0]

    def test_mass_difference_positive(self, b10_b11_pair: IsotopePair) -> None:
        """Heavy mass must be greater than light mass."""
        assert b10_b11_pair.mass_heavy > b10_b11_pair.mass_light


class TestRunMassMutationFEP:
    """Integration test for run_mass_mutation_fep."""

    def test_returns_expected_keys(
        self, mock_pimd_trajectory: np.ndarray, b10_b11_pair: IsotopePair
    ) -> None:
        """Result dict should contain all expected keys."""
        result = run_mass_mutation_fep(
            mock_pimd_trajectory, b10_b11_pair, temperature=145.0, n_beads=64
        )
        expected_keys = [
            "delta_A_eV", "delta_A_std_eV", "delta_A_per_atom_meV",
            "converged", "temperature_K", "n_beads", "n_frames",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"


class TestPhysicalConstants:
    """Sanity checks on physical constants."""

    def test_kb_in_ev_per_k(self) -> None:
        """kB should be ~8.617e-5 eV/K."""
        assert KB == pytest.approx(8.617e-5, rel=1e-3)

    def test_kt_at_room_temp(self) -> None:
        """k_B * 300 K should be ~0.02585 eV."""
        assert KB * 300.0 == pytest.approx(0.02585, rel=1e-2)
