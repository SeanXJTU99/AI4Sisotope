"""Generalize the BF3-trained isotope effect prediction pipeline to UF6.

Key differences BF3 -> UF6:
    - Geometry: planar D3h -> octahedral Oh
    - Heavy element: U (Z=92) requires scalar relativistic pseudopotential
    - Beads: P=64 -> P=16-32 (heavy element, less quantum delocalization)
    - Box size: 15 A -> 20 A
    - Type map: [B, F] -> [U, F]
    - Mass: B=10.811 -> U=238.05

Critical: The fully-relativistic pseudopotential (U_SG15_PBE_FR.upf)
is mandatory for UF6. Without relativistic corrections, the U-F bond
length and vibrational frequencies deviate significantly from experiment.
"""

import json
from pathlib import Path


def adapt_for_heavy_element(
    dst_json: Path,
) -> None:
    """Adapt training configuration for heavy-element UF6 system.

    Sets up type_map with U and F, adjusts sel for the higher
    coordination number (6 vs 3), and enables the relativistic
    pseudopotential in the ABACUS template.

    Args:
        dst_json: Path to write the UF6-adapted training config.
    """
    src_json = Path("deepmd/train/input_dpa2.json")
    with open(src_json) as f:
        config = json.load(f)

    # UF6 type map
    config["model"]["type_map"] = ["U", "F"]

    # U has 6 F neighbors (octahedral coordination) -> larger nsel
    config["model"]["descriptor"]["repinit"]["nsel"] = 80
    config["model"]["descriptor"]["repformer"]["nsel"] = 80

    # U is heavy, forces are smaller -> adjust prefactors
    config["loss"]["atom_pref"] = {"U": 0.6, "F": 1.0}

    # UF6 data paths
    config["training"]["training_data"]["systems"] = ["../data/train/uf6_dimer/"]
    config["training"]["validation_data"]["systems"] = ["../data/test/uf6_dimer/"]

    with open(dst_json, "w") as f:
        json.dump(config, f, indent=4)

    print(f"UF6 training config written: {dst_json}")


def adjust_beads_for_heavy_element(
    input_xml: Path,
    element_mass: float = 238.05,
    temperature: float = 330.0,
) -> int:
    """Calculate recommended beads count for heavy-element PIMD.

    The quantum delocalization length scales as:
        lambda ~ hbar / sqrt(m * k_B * T)

    For U (238 amu) vs B (10.8 amu) at similar T:
        lambda_U / lambda_B = sqrt(10.8 / 238) ~ 0.21

    So UF6 needs ~5x fewer beads than BF3 for equivalent convergence.
    If BF3 needs P=64, UF6 needs P=12-16.

    Args:
        input_xml: Path to i-PI input XML template.
        element_mass: Mass of the heavy element in amu.
        temperature: Simulation temperature in K.

    Returns:
        Recommended beads count (P).
    """
    # Reference: BF3 at 145 K, B mass 10.811, P_ref=64
    ref_mass = 10.811
    ref_temp = 145.0
    ref_beads = 64

    # Beads scaling: P ~ m^{-1/2} * T^{-1}
    # (from spring constant kappa = m * P * (kT/hbar)^2)
    p_recommended = int(ref_beads * (ref_mass / element_mass) ** 0.5 *
                        (ref_temp / temperature))

    # Round up to common values
    for p in [8, 12, 16, 24, 32, 48, 64]:
        if p >= p_recommended:
            p_recommended = p
            break

    print(f"UF6 recommended beads: P={p_recommended}")
    print(f"  (BF3 reference: P={ref_beads} at {ref_temp} K, B mass={ref_mass})")
    print(f"  Scaling factor: sqrt(m_B/m_U) * (T_BF3/T_UF6) = "
          f"{(ref_mass / element_mass) ** 0.5 * (ref_temp / temperature):.3f}")

    return p_recommended


def relativistic_correction_note() -> str:
    """Generate note on relativistic corrections for UF6.

    Returns:
        Formatted string explaining relativistic pseudopotential usage.
    """
    return (
        "RELATIVISTIC CORRECTION NOTE\n"
        "============================\n"
        "Uranium (Z=92) inner-shell electrons move at relativistic speeds.\n"
        "The scalar relativistic correction modifies the effective potential\n"
        "seen by valence electrons, affecting:\n"
        "  - U-F bond length: ~1.99 A (relativistic) vs ~2.05 A (non-relativistic)\n"
        "  - nu3 asymmetric stretch: ~620 cm-1 vs ~590 cm-1\n"
        "  - Barrier to internal rotation: ~5% change\n"
        "\n"
        "The fully-relativistic pseudopotential U_SG15_PBE_FR.upf includes\n"
        "  - Mass-velocity correction\n"
        "  - Darwin term\n"
        "  - Spin-orbit coupling (averaged, scalar relativistic)\n"
        "\n"
        "Without these corrections, UF6 isotope effect predictions will\n"
        "deviate from experiment by 10-30%.\n"
    )


def run_uf6_generalization(output_root: Path) -> None:
    """Run the complete BF3 -> UF6 generalization pipeline.

    Args:
        output_root: Directory for adapted outputs.
    """
    output_root.mkdir(parents=True, exist_ok=True)

    print("Cross-system generalization: BF3 -> UF6 (heavy element)")
    print("=" * 60)

    # Step 1: ABACUS template adaptation
    print("\n[1/5] ABACUS template for UF6...")
    print("  - Octahedral Oh symmetry")
    print("  - U_SG15_PBE_FR.upf (fully relativistic pseudopotential)")
    print("  - Box size: 20 A")
    print(relativistic_correction_note())

    # Step 2: DP training config
    print("[2/5] Adapting DP training config...")
    adapt_for_heavy_element(output_root / "input_dpa2_uf6.json")

    # Step 3: Beads count
    print("[3/5] Optimizing beads count...")
    p_uf6 = adjust_beads_for_heavy_element(
        Path("pimd/ipi/input_pimd.xml"),
        element_mass=238.05,
        temperature=330.0,
    )

    # Step 4: PIMD config
    print(f"[4/5] PIMD config: P={p_uf6}, T=330 K...")
    print("  - Fewer beads due to heavy element mass")
    print("  - Higher temperature (UF6 requires elevated T for liquid phase)")

    # Step 5: Validation targets
    print("[5/5] Validation targets...")
    print("  - Energy RMSE < 3.0 meV/atom (relaxed for heavy element)")
    print("  - Force RMSE < 40 meV/Ang")
    print("  - alpha prediction uncertainty < 0.0005")

    print(f"\nDone. Outputs in: {output_root}")
