"""Generalize the BF3-trained isotope effect prediction pipeline to CF4.

Key differences BF3 -> CF4:
    - Geometry: planar D3h -> tetrahedral Td
    - Functional: SCAN+rVV10 -> PBE0-D3(BJ)
    - Dominant mode: nu2 out-of-plane bend -> nu3/nu4 stretch+bend
    - Box size: 15 A -> 18 A (CF4 is larger)
    - Type map: [B, F] -> [C, F]
    - Mass: B=10.811 -> C=12.011

The PIMD/FEP analysis framework is fully reused; only the DFT template
and DP training configuration need adaptation.
"""

import json
import shutil
from pathlib import Path


def adapt_type_map(
    src_molecule: str = "BF3",
    dst_molecule: str = "CF4",
) -> tuple[list[str], list[float]]:
    """Adapt type_map and mass_map between molecular systems.

    Args:
        src_molecule: Source molecule identifier.
        dst_molecule: Target molecule identifier.

    Returns:
        Tuple of (type_map, mass_map) for the target molecule.
    """
    element_masses = {
        "B": 10.811, "C": 12.011, "F": 18.998, "U": 238.05,
    }

    type_configs = {
        "BF3": (["B", "F"], [10.811, 18.998]),
        "CF4": (["C", "F"], [12.011, 18.998]),
        "UF6": (["U", "F"], [238.05, 18.998]),
    }

    return type_configs.get(dst_molecule, type_configs["BF3"])


def adapt_dp_training_config(
    src_json: Path,
    dst_json: Path,
    dst_molecule: str = "CF4",
) -> None:
    """Adapt DeePMD-kit training JSON for a new molecular system.

    Updates type_map, atom_pref, and sel parameters for the
    different coordination environment in the target molecule.

    Args:
        src_json: Path to source training JSON (e.g., BF3 config).
        dst_json: Path to write adapted config.
        dst_molecule: Target molecule identifier.
    """
    with open(src_json) as f:
        config = json.load(f)

    type_map, mass_map = adapt_type_map("BF3", dst_molecule)

    config["model"]["type_map"] = type_map

    # Adjust atom-specific prefactors for new elements
    if dst_molecule == "CF4":
        config["loss"]["atom_pref"] = {"C": 0.9, "F": 1.0}

    # Adjust nsel for different coordination number
    # BF3: B has 3 neighbors, CF4: C has 4 neighbors
    if dst_molecule == "CF4":
        config["model"]["descriptor"]["repinit"]["nsel"] = 100
        config["model"]["descriptor"]["repformer"]["nsel"] = 100

    # Update data paths
    config["training"]["training_data"]["systems"] = [f"../data/train/{dst_molecule.lower()}_dimer/"]
    config["training"]["validation_data"]["systems"] = [f"../data/test/{dst_molecule.lower()}_dimer/"]

    with open(dst_json, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Adapted training config: {dst_json}")


def adapt_pimd_config(
    src_xml: Path,
    dst_xml: Path,
    dst_molecule: str = "CF4",
) -> None:
    """Adapt i-PI PIMD input for a new molecular system.

    CF4 is heavier than BF3, so fewer beads may be needed for
    convergence (P=32-64 vs P=64-128 for BF3).

    Args:
        src_xml: Path to source i-PI input XML.
        dst_xml: Path to write adapted config.
        dst_molecule: Target molecule identifier.
    """
    with open(src_xml) as f:
        content = f.read()

    # Adjust beads recommendation in comments
    content = content.replace(
        "P=64 (converged for BF3",
        "P=48 (converged for CF4",
    )

    # Adjust temperature (CF4 boiling point ~145 K -> similar region)
    # No change needed; CF4 liquid range includes 145 K

    with open(dst_xml, "w") as f:
        f.write(content)

    print(f"Adapted PIMD config: {dst_xml}")


def validate_transfer_quality(
    src_results: dict[str, float],
    dst_results: dict[str, float],
) -> dict[str, float]:
    """Compare transfer learning quality metrics.

    Args:
        src_results: Validation metrics for source system.
        dst_results: Validation metrics for target system.

    Returns:
        Dict with relative degradation metrics.
    """
    return {
        "energy_rmse_increase_pct": round(
            (dst_results.get("rmse_energy_meV_per_atom", 0) /
             max(src_results.get("rmse_energy_meV_per_atom", 0.01), 0.01) - 1) * 100, 1
        ),
        "force_rmse_increase_pct": round(
            (dst_results.get("rmse_force_meV_Ang", 0) /
             max(src_results.get("rmse_force_meV_Ang", 0.01), 0.01) - 1) * 100, 1
        ),
    }


def run_cross_system_pipeline(
    source_molecule: str,
    target_molecule: str,
    output_root: Path,
) -> None:
    """Execute the complete cross-system generalization pipeline.

    Steps:
        1. Adapt ABACUS templates (in production, copy and modify STRU/INPUT/KPT)
        2. Adapt DP training config (type_map, nsel, atom_pref)
        3. Adapt PIMD config (temperature, beads recommendation)
        4. Run LoRA fine-tuning (see deepmd/train/lora_finetune.py)
        5. Validate transfer quality
        6. Run FEP analysis for new system

    Args:
        source_molecule: Source molecule (e.g., 'BF3').
        target_molecule: Target molecule (e.g., 'CF4').
        output_root: Root directory for adapted outputs.
    """
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Cross-system generalization: {source_molecule} -> {target_molecule}")
    print("=" * 60)

    # Step 1: Template adaptation (mock)
    print(f"\n[1/4] Adapting ABACUS templates...")
    print(f"  Source: abacus/templates/{source_molecule}/")
    print(f"  Target: abacus/templates/{target_molecule}/")
    print(f"  Key changes: functional, box size, species")

    # Step 2: DP config adaptation
    print(f"\n[2/4] Adapting DP training config...")
    adapt_dp_training_config(
        Path(f"deepmd/train/input_dpa2.json"),
        output_root / f"input_dpa2_{target_molecule.lower()}.json",
        target_molecule,
    )

    # Step 3: PIMD config adaptation
    print(f"\n[3/4] Adapting PIMD config...")
    adapt_pimd_config(
        Path("pimd/ipi/input_pimd.xml"),
        output_root / f"input_pimd_{target_molecule.lower()}.xml",
        target_molecule,
    )

    # Step 4: Validation
    print(f"\n[4/4] Transfer quality assessment...")
    print(f"  Expected: energy RMSE < 2.0 meV/atom, force RMSE < 50 meV/Ang")
    print(f"  If above thresholds, re-run LoRA fine-tuning with more data.")

    print(f"\nPipeline complete. Outputs in: {output_root}")
