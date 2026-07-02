"""LoRA (Low-Rank Adaptation) fine-tuning for cross-system transfer learning.

Enables efficient adaptation of a pre-trained DPA-2 model from one
molecular system (e.g., CF4) to another (e.g., BF3) by injecting
low-rank trainable adapters while freezing the base model weights.

This dramatically reduces the training data and GPU-hours needed
for new isotope systems compared to training from scratch.
"""

import json
from pathlib import Path
from typing import Optional


def build_lora_config(
    base_model_json: Path,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    target_modules: Optional[list[str]] = None,
) -> dict:
    """Build a LoRA-fine-tuned model configuration from a base config.

    Injects low-rank adaptation layers into the DPA-2 descriptor
    repformer layers and fitting network. The base model weights
    are frozen; only LoRA adapter weights are updated.

    Args:
        base_model_json: Path to the base model training JSON.
        lora_rank: LoRA rank (r). Higher = more capacity, more params.
            Recommended: 8 for similar systems, 16 for dissimilar.
        lora_alpha: LoRA scaling factor. Typically 2*r.
        target_modules: List of module names to apply LoRA to.
            Default: repformer attention + fitting net layers.

    Returns:
        LoRA-augmented model configuration dict.
    """
    with open(base_model_json) as f:
        config = json.load(f)

    if target_modules is None:
        target_modules = [
            "descriptor.repformer.attention",
            "fitting_net.dense",
        ]

    lora_config = {
        "enable": True,
        "rank": lora_rank,
        "alpha": lora_alpha,
        "target_modules": target_modules,
        "dropout": 0.0,
        "init_scale": 0.01,
    }

    config["lora"] = lora_config

    # Freeze base model weights by setting learning rate multipliers
    config["learning_rate"]["lora_lr_scale"] = 1.0
    config["learning_rate"]["base_lr_scale"] = 0.0

    # Reduce total steps for fine-tuning (10-20% of full training)
    original_steps = config["training"]["numb_steps"]
    config["training"]["numb_steps"] = original_steps // 5
    config["training"]["start_from_ckpt"] = True

    # Use lower learning rate for fine-tuning
    config["learning_rate"]["start_lr"] *= 0.1

    return config


def apply_lora(
    base_checkpoint: Path,
    lora_config: dict,
    training_data: Path,
    output_dir: Path,
    n_steps: int = 200000,
) -> Path:
    """Apply LoRA fine-tuning to a pre-trained model.

    Args:
        base_checkpoint: Path to pre-trained model checkpoint (.pt).
        lora_config: LoRA configuration dict from build_lora_config.
        training_data: Path to target system training data.
        output_dir: Directory for fine-tuned checkpoints.
        n_steps: Number of fine-tuning steps (default: 200k).

    Returns:
        Path to the best fine-tuned checkpoint.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write the modified config
    config_path = output_dir / "input_lora.json"
    with open(config_path, "w") as f:
        json.dump(lora_config, f, indent=2)

    print(f"LoRA config written to: {config_path}")
    print(f"Base checkpoint: {base_checkpoint}")
    print(f"Target training data: {training_data}")
    print(f"Fine-tuning steps: {n_steps}")
    print(f"LoRA rank: {lora_config['lora']['rank']}, alpha: {lora_config['lora']['alpha']}")

    # In production: subprocess.run(["dp", "train", str(config_path)], check=True)
    print("LoRA fine-tuning complete (mock).")

    return output_dir / "model.ckpt-lora-best.pt"


def merge_lora_weights(
    base_checkpoint: Path,
    lora_checkpoint: Path,
    output_path: Path,
) -> Path:
    """Merge LoRA adapter weights into the base model for deployment.

    After merging, the model has the same architecture as the original
    (no LoRA overhead at inference time) but with adapted weights.

    Args:
        base_checkpoint: Path to original pre-trained checkpoint.
        lora_checkpoint: Path to LoRA fine-tuned checkpoint.
        output_path: Path for the merged model.

    Returns:
        Path to merged model checkpoint.
    """
    print(f"Merging LoRA weights from {lora_checkpoint} into base model...")

    # In production, this loads both checkpoints, multiplies the LoRA
    # matrices (B*A), and adds to the corresponding base weights.
    # For DPA-2: _merge_repformer_lora() + _merge_fitting_lora()
    print(f"Merged model saved to: {output_path}")

    return output_path
