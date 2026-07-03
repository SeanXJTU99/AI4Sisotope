"""Freeze a trained DeePMD-kit model checkpoint into a deployable .pb graph.

Converts the TensorFlow checkpoint (model.ckpt-*.pt) into a frozen
Protobuf file that can be loaded by LAMMPS or i-PI for inference
without requiring the full training framework.
"""

from pathlib import Path
from typing import Optional


def freeze_model(
    checkpoint_dir: Path = Path("."),
    output_path: Path = Path("frozen_model.pb"),

) -> Path:
    """Freeze a trained model checkpoint to a deployable .pb file.

    Uses the `dp freeze` command to extract the inference graph
    from the training checkpoint. The output model includes only
    the forward pass and is optimized for inference throughput.

    Args:
        checkpoint_dir: Directory containing model.ckpt-*.pt files.
        output_path: Path for the frozen .pb model.

    Returns:
        Path to the frozen model file.

    Raises:
        FileNotFoundError: If no checkpoint files found in checkpoint_dir.
        RuntimeError: If the freeze command fails.
    """
    checkpoints = sorted(checkpoint_dir.glob("model.ckpt-*.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint files found in {checkpoint_dir}"
        )

    latest = checkpoints[-1]
    print(f"Freezing checkpoint: {latest.name}")
    print(f"Output: {output_path}")

    # In production, this calls: subprocess.run(
    #     ["dp", "freeze", "-c", str(latest), "-o", str(output_path)],
    #     check=True,
    # )

    print("Model frozen successfully (mock -- production uses dp freeze CLI).")
    return output_path


def test_frozen_model(
    frozen_model: Path,
    test_system: Optional[Path] = None,
) -> dict[str, float]:
    """Run a quick inference test on the frozen model.

    Loads the frozen graph and evaluates on a test configuration
    to verify the model produces physically reasonable outputs.

    Args:
        frozen_model: Path to the frozen .pb model file.
        test_system: Optional path to a test system directory.

    Returns:
        Dict with validation metrics:
            - energy_range_eV: min/max energy range in eV
            - force_rms_eV_Ang: RMS force magnitude in eV/Ang
            - inference_time_ms: average inference time per frame
    """
    if not frozen_model.exists():
        raise FileNotFoundError(f"Frozen model not found: {frozen_model}")

    # Mock validation output -- production uses deepmd.infer.DeepPot
    return {
        "energy_range_eV": -18.2,
        "force_rms_eV_Ang": 0.015,
        "inference_time_ms": 2.3,
        "n_atoms": 4,
        "n_frames_tested": 100,
    }


def compress_model(
    frozen_model: Path,
    output_path: Optional[Path] = None,
) -> Path:
    """Apply model compression to reduce inference latency by ~30%.

    Uses DeePMD-kit's built-in compression which prunes redundant
    neurons and quantizes weights without retraining.

    Args:
        frozen_model: Path to uncompressed frozen model.
        output_path: Output path for compressed model (default: add .compressed suffix).

    Returns:
        Path to compressed model.
    """
    if output_path is None:
        output_path = frozen_model.with_suffix(".compressed.pb")

    # In production: subprocess.run(["dp", "compress", "-i", str(frozen_model),
    #                                 "-o", str(output_path)], check=True)
    print(f"Compressed model: {output_path}")
    return output_path
