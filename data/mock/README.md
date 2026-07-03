# Mock Data for Demonstration

## Purpose

All data files in this directory are mock (synthetic) datasets generated
for format demonstration and pipeline testing. They do **not** contain
real experimental or DFT-calculated values.

## File Formats

### bf3_dimer.xyz
- Format: Extended XYZ (atom count, comment line, element + xyz per atom)
- System: Two BF3 molecules at ~3.5 A separation
- Coordinates: Perturbed from ideal D3h geometry via `np.random.normal(0, 0.1)`
- Usage: Initial configuration for DP-GEN exploration and PIMD initialization

### Expected Production Data Formats (not included)

```
data/train/bf3_dimer/
  set.000/
    box.npy       -- (n_frames, 9) unit cell vectors in Angstrom
    coord.npy     -- (n_frames, n_atoms * 3) atomic positions in Angstrom
    energy.npy    -- (n_frames,) total energy in eV
    force.npy     -- (n_frames, n_atoms * 3) atomic forces in eV/Ang
    virial.npy    -- (n_frames, 9) virial tensor in eV
  type_map.raw    -- element names (one per line)
  type.raw         -- atom type indices (n_atoms integers)
```

## Generating Your Own Data

1. Place real DFT calculation results following the DeePMD-kit Type-Raw format
2. Run `abacus/scripts/extract_data.py` to convert ABACUS output
3. Use `dp_gen/run_dpgen.sh` to run the active learning pipeline
4. Training data quality: 50,000+ frames from DP-GEN for production models

## Seed Reproducibility

All mock data uses `np.random.default_rng(42)` for deterministic generation.
Change the seed to generate different mock configurations.
