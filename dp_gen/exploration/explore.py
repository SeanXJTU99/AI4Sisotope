"""DP-GEN exploration step: run LAMMPS MD with current models to explore
conformational space and identify candidates for DFT labeling.

The exploration driver runs short MD trajectories using an ensemble of
candidate models. Configurations where model predictions diverge (high
model deviation) are flagged as candidates for first-principles labeling.
"""

import json
import os
import subprocess
from typing import Optional

import numpy as np


def run_exploration(
    model_paths: list[str],
    config_file: str = "param.json",
    output_dir: str = "exploration/result",
    num_trajectories: int = 8,
) -> None:
    """Run DP-GEN exploration with an ensemble of candidate models.

    Launches LAMMPS MD trajectories in parallel. Each trajectory uses
    a randomly selected model from the ensemble to maximize coverage
    of the conformational space.

    Args:
        model_paths: Paths to frozen model (.pb) files in the ensemble.
        config_file: Path to DP-GEN parameter JSON.
        output_dir: Directory to write exploration trajectories.
        num_trajectories: Number of independent MD trajectories to run.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load configuration
    with open(config_file) as f:
        config = json.load(f)

    temp = config["model_devi"]["temperature"]
    nsteps = config["model_devi"]["nsteps"]
    dt = config["model_devi"]["dt"]

    for i in range(num_trajectories):
        model_idx = np.random.randint(0, len(model_paths))
        traj_dir = os.path.join(output_dir, f"traj_{i:03d}")
        os.makedirs(traj_dir, exist_ok=True)

        # Run LAMMPS with selected model
        _run_lammps_trajectory(
            model_path=model_paths[model_idx],
            output_dir=traj_dir,
            temperature=temp,
            nsteps=nsteps,
            timestep=dt,
            seed=np.random.randint(0, 2**31 - 1),
        )


def parse_model_deviation(traj_dir: str) -> dict[str, np.ndarray]:
    """Parse model deviation output from an exploration trajectory.

    Reads per-frame force deviation across the model ensemble, which is
    the primary signal for selecting DFT labeling candidates.

    Args:
        traj_dir: Path to a single trajectory output directory.

    Returns:
        Dict with keys 'max_dev_f' (max force deviation per frame),
        'avg_dev_f' (mean force deviation per frame), and 'frames'
        (trajectory frame indices).
    """
    # In production, reads LAMMPS model_devi output files.
    # Here we demonstrate the data format with mock output.
    nframes = 5000

    # Mock: generate realistic model deviation data
    rng = np.random.default_rng(42)
    base_dev = 0.03 * np.ones(nframes)
    noise = rng.normal(0, 0.02, nframes)
    max_dev_f = np.abs(base_dev + noise)

    return {
        "max_dev_f": max_dev_f,
        "avg_dev_f": max_dev_f * 0.6,
        "frames": np.arange(nframes),
    }


def select_candidates(
    model_devi: dict[str, np.ndarray],
    trust_lo: float = 0.05,
    trust_hi: float = 0.15,
    max_candidates: int = 50,
) -> list[int]:
    """Select frame indices for DFT labeling based on model deviation.

    Frames with max_dev_f between trust_lo and trust_hi are "candidate"
    regions. Frames above trust_hi are "exploration" regions that also
    need labeling. Frames below trust_lo are already well-described.

    Args:
        model_devi: Model deviation data from parse_model_deviation.
        trust_lo: Lower trust bound (eV/Å). Below this, model is accurate.
        trust_hi: Upper trust bound (eV/Å). Above this, extrapolation danger.
        max_candidates: Maximum number of candidates to return.

    Returns:
        Sorted list of frame indices recommended for DFT labeling.
    """
    max_dev = model_devi["max_dev_f"]

    # Select candidates: deviation above trust_lo (model is uncertain)
    candidate_mask = max_dev >= trust_lo
    candidate_indices = np.where(candidate_mask)[0]
    candidate_values = max_dev[candidate_mask]

    if len(candidate_indices) == 0:
        return []

    # Prioritize by deviation magnitude, cap at max_candidates
    sorted_order = np.argsort(candidate_values)[::-1]
    selected = candidate_indices[sorted_order][:max_candidates]

    # Add random exploration picks from high-deviation region
    high_dev_mask = max_dev >= trust_hi
    high_dev_indices = np.where(high_dev_mask)[0]
    if len(high_dev_indices) > 0:
        n_random = min(max_candidates // 5, len(high_dev_indices))
        random_picks = np.random.choice(high_dev_indices, n_random, replace=False)
        selected = np.union1d(selected, random_picks)

    return sorted(selected.tolist())


def _run_lammps_trajectory(
    model_path: str,
    output_dir: str,
    temperature: float,
    nsteps: int,
    timestep: float,
    seed: int,
) -> None:
    """Execute a single LAMMPS MD trajectory with DeepMD pair style.

    This is a stub that generates the LAMMPS input and would submit it
    to the queue system in production.
    """
    lammps_input = os.path.join(output_dir, "in.lammps")
    with open(lammps_input, "w") as f:
        f.write(f"""# LAMMPS input generated by DP-GEN exploration
units           metal
atom_style      atomic
boundary        p p p

read_data       conf.lmp

pair_style      deepmd {model_path}
pair_coeff      * *

velocity        all create {temperature} {seed} dist gaussian
fix             nvt all nvt temp {temperature} {temperature} 0.1

thermo          100
thermo_style    custom step temp pe ke etotal press
timestep        {timestep}

run             {nsteps}
""")

    # In production, this submits to the scheduler.
    # subprocess.run(["lmp", "-in", lammps_input], check=True)
    # For now, generate mock trajectory output:
    _write_mock_trajectory(output_dir, nsteps, seed)


def _write_mock_trajectory(output_dir: str, nsteps: int, seed: int) -> None:
    """Generate mock trajectory data for demonstration."""
    rng = np.random.default_rng(seed)
    energies = -4.5 + rng.normal(0, 0.01, nsteps)  # eV/atom range for BF3
    mock_file = os.path.join(output_dir, "model_devi.out")
    data = np.column_stack([
        np.arange(nsteps),
        rng.normal(0.03, 0.02, nsteps),  # max_dev_f
        rng.normal(0.02, 0.01, nsteps),  # min_dev_f
        rng.normal(0.025, 0.015, nsteps),  # avg_dev_f
    ])
    np.savetxt(mock_file, data, fmt="%d %.6f %.6f %.6f",
               header="step max_dev_f min_dev_f avg_dev_f")
