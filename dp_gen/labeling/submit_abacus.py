"""Submit selected candidate configurations for ABACUS single-point
energy calculations during the DP-GEN labeling phase.

Reads candidate frame indices from the exploration step, prepares
ABACUS input files from templates, submits jobs, and collects results
into the DP-GEN training data format.
"""

import json
import os
import shutil
from typing import Optional

import numpy as np


# --- Physical constants preserved as-is (real production values) ---
SCF_THR = 1e-7          # eV — tight SCF convergence for force accuracy
SCF_NMAX = 200           # maximum SCF iterations
FORCE_THR_EV_ANG = 0.01  # eV/Å — force convergence criterion


def prepare_abacus_inputs(
    candidate_indices: list[int],
    template_dir: str = "abacus/templates/BF3",
    output_dir: str = "labeling/tasks",
    box_size: float = 15.0,
) -> list[str]:
    """Prepare ABACUS input files for each candidate configuration.

    Reads STRU, INPUT, and KPT templates, substitutes the candidate
    geometry (with random perturbation for data anonymization), and
    writes complete ABACUS input directories.

    Args:
        candidate_indices: Frame indices selected for DFT labeling.
        template_dir: Path to ABACUS template directory.
        output_dir: Root directory for labeling tasks.
        box_size: Cubic box side length in Ångström.

    Returns:
        List of task directory paths ready for ABACUS execution.
    """
    os.makedirs(output_dir, exist_ok=True)

    stru_template = _read_template(os.path.join(template_dir, "STRU.template"))
    input_template = _read_template(os.path.join(template_dir, "INPUT.template"))
    kpt_template = _read_template(os.path.join(template_dir, "KPT.template"))

    task_dirs: list[str] = []

    for idx in candidate_indices:
        task_dir = os.path.join(output_dir, f"task_{idx:06d}")
        os.makedirs(task_dir, exist_ok=True)

        # Generate perturbed geometry (data anonymization)
        stru_content = _fill_stru_template(stru_template, box_size, seed=idx)

        with open(os.path.join(task_dir, "STRU"), "w") as f:
            f.write(stru_content)
        # INPUT and KPT are preserved verbatim (functional parameters)
        with open(os.path.join(task_dir, "INPUT"), "w") as f:
            f.write(input_template)
        with open(os.path.join(task_dir, "KPT"), "w") as f:
            f.write(kpt_template)

        task_dirs.append(task_dir)

    return task_dirs


def submit_jobs(
    task_dirs: list[str],
    scheduler: str = "local",
    nprocs: int = 16,
) -> list[int]:
    """Submit ABACUS calculation jobs to the scheduler.

    In production, this dispatches to Slurm or PBS. For local execution,
    runs ABACUS directly with mpirun.

    Args:
        task_dirs: List of ABACUS input directories.
        scheduler: Job scheduler type ("local", "slurm", "pbs").
        nprocs: Number of MPI processes per task.

    Returns:
        List of job IDs for tracking.
    """
    job_ids: list[int] = []

    for i, task_dir in enumerate(task_dirs):
        # Mock: simulate successful submission
        job_id = 10000 + i
        job_ids.append(job_id)

        # In production:
        # if scheduler == "slurm":
        #     subprocess.run(["sbatch", f"--ntasks={nprocs}",
        #                     f"--job-name=abacus_{i}", "run_abacus.sh"],
        #                    cwd=task_dir)
        # else:
        #     subprocess.run(["mpirun", "-np", str(nprocs), "abacus"],
        #                    cwd=task_dir)

        # Generate mock ABACUS output for demonstration
        _write_mock_output(task_dir, job_id)

    return job_ids


def collect_results(
    task_dirs: list[str],
    output_file: str = "labeling/result/training_data",
) -> dict[str, int]:
    """Collect ABACUS results and compile into DeePMD-kit training format.

    Parses ABACUS output files (energy, forces, virial) and packages
    them as .npy sets for DP-GEN training.

    Args:
        task_dirs: List of completed ABACUS task directories.
        output_file: Base path for output training data files.

    Returns:
        Summary dict with counts of successful/failed/converged tasks.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    n_total = len(task_dirs)
    n_converged = 0
    n_failed = 0

    all_energies: list[float] = []
    all_forces: list[np.ndarray] = []
    all_virials: list[np.ndarray] = []
    all_coords: list[np.ndarray] = []
    all_types: list[np.ndarray] = []

    for task_dir in task_dirs:
        try:
            energy, forces, virial, coords, atom_types = _parse_abacus_output(task_dir)
            all_energies.append(energy)
            all_forces.append(forces)
            all_virials.append(virial)
            all_coords.append(coords)
            all_types.append(atom_types)
            n_converged += 1
        except (FileNotFoundError, ValueError):
            n_failed += 1
            continue

    if n_converged > 0:
        # Save in DeePMD-kit set.* format
        np.save(f"{output_file}/set.000/energy.npy",
                np.array(all_energies, dtype=np.float64))
        np.save(f"{output_file}/set.000/force.npy",
                np.stack(all_forces).astype(np.float64))
        np.save(f"{output_file}/set.000/virial.npy",
                np.stack(all_virials).astype(np.float64))
        np.save(f"{output_file}/set.000/coord.npy",
                np.stack(all_coords).astype(np.float64))
        np.save(f"{output_file}/set.000/type_map.raw",
                all_types[0].astype(np.int32))

    return {
        "total": n_total,
        "converged": n_converged,
        "failed": n_failed,
    }


def _read_template(path: str) -> str:
    """Read a template file, returning its content as a string."""
    with open(path) as f:
        return f.read()


def _fill_stru_template(template: str, box_size: float, seed: int) -> str:
    """Fill STRU template with perturbed atomic coordinates.

    Applies random normal perturbation to anonymize coordinates while
    preserving atom types and lattice vectors.

    Args:
        template: STRU template content.
        box_size: Cubic box side length in Ångström.
        seed: Random seed for reproducible perturbation.

    Returns:
        Filled STRU content with perturbed coordinates.
    """
    rng = np.random.default_rng(seed)
    perturbation = rng.normal(0, 0.1, size=(4, 3))

    lines: list[str] = []
    in_coords = False
    atom_idx = 0

    for line in template.splitlines():
        if "ATOMIC_POSITIONS" in line:
            in_coords = True
            lines.append(line)
            continue
        if in_coords and line.strip() and not line.startswith("#"):
            parts = line.split()
            if len(parts) >= 4:
                element = parts[0]
                px, py, pz = float(parts[1]), float(parts[2]), float(parts[3])
                if atom_idx < len(perturbation):
                    px += perturbation[atom_idx][0]
                    py += perturbation[atom_idx][1]
                    pz += perturbation[atom_idx][2]
                    atom_idx += 1
                lines.append(f"{element}  {px:.10f}  {py:.10f}  {pz:.10f}  1 1 1")
                continue
        in_coords = False
        lines.append(line)

    return "\n".join(lines)


def _parse_abacus_output(task_dir: str) -> tuple[
    float, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Parse ABACUS output files for energy, forces, virial, and structure.

    Reads the standard ABACUS output files and extracts training data.
    In production, reads STRU_ION_D, running_*.log, etc.
    """
    rng = np.random.default_rng(hash(task_dir) % (2**31))

    # Mock: realistic energy range for BF3 with SCAN+rVV10 (~ -4.5 eV/atom)
    natoms = 4
    energy = float(-4.5 * natoms + rng.normal(0, 0.005))
    forces = np.array(rng.normal(0, 0.02, (natoms, 3)), dtype=np.float64)
    virial = np.array(rng.normal(0, 0.1, (9,)), dtype=np.float64)
    coords = np.array(rng.normal(0, 0.1, (natoms, 3)), dtype=np.float64)
    atom_types = np.array([0, 1, 1, 1], dtype=np.int32)

    return energy, forces, virial, coords, atom_types


def _write_mock_output(task_dir: str, job_id: int) -> None:
    """Generate mock ABACUS output files for demonstration."""
    rng = np.random.default_rng(job_id)
    natoms = 4
    energy_per_atom = -4.5 + rng.normal(0, 0.003)

    # Mock STRU_ION_D (final geometry)
    with open(os.path.join(task_dir, "STRU_ION_D"), "w") as f:
        f.write("ATOMIC_SPECIES\nB 10.811 B_ONCV_PBE-1.0.upf\nF 18.998 F_ONCV_PBE-1.0.upf\n\n")
        f.write("LATTICE_VECTORS\n15.0 0.0 0.0\n0.0 15.0 0.0\n0.0 0.0 15.0\n\n")
        f.write("ATOMIC_POSITIONS\nDirect\n\n")
        f.write("B 0.5000 0.5000 0.5000 1 1 1\n")
        f.write("F 0.5200 0.5000 0.5000 1 1 1\n")
        f.write("F 0.4800 0.4800 0.5000 1 1 1\n")
        f.write("F 0.4800 0.5200 0.4800 1 1 1\n")

    # Mock running log (SCF convergence)
    with open(os.path.join(task_dir, "running_scf.log"), "w") as f:
        f.write(f"FINAL_ETOT_IS {energy_per_atom * natoms:.10f} eV\n")
        f.write("SCF convergence achieved in 18 iterations\n")
        f.write(f"Total force RMS: {rng.normal(0.005, 0.002):.6f} eV/Ang\n")
